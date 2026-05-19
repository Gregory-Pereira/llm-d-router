/*
Copyright 2026 The llm-d Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"

	. "github.com/onsi/ginkgo/v2" // nolint:revive
	. "github.com/onsi/gomega"    // nolint:revive

	"github.com/llm-d/llm-d-router/pkg/common/routing"
	"github.com/llm-d/llm-d-router/test/sidecar/mock"
)

type mooncakeTestInfo struct {
	sidecarTestInfo
	bootstrapHandler *mock.MooncakeBootstrapHandler
}

func mooncakeTestSetup() *mooncakeTestInfo {
	testInfo := &mooncakeTestInfo{}

	testInfo.ctx = newTestContext()
	testInfo.ctx, testInfo.cancelFn = context.WithCancel(testInfo.ctx)
	testInfo.stoppedCh = make(chan struct{})

	// Decoder
	testInfo.decodeHandler = &mock.ChatCompletionHandler{
		Connector: KVConnectorMooncake,
		Role:      mock.RoleDecode,
	}
	testInfo.decodeBackend = httptest.NewServer(testInfo.decodeHandler)
	DeferCleanup(testInfo.decodeBackend.Close)

	// Prefiller with bootstrap /query endpoint
	testInfo.prefillHandler = &mock.ChatCompletionHandler{
		Connector: KVConnectorMooncake,
		Role:      mock.RolePrefill,
	}
	testInfo.bootstrapHandler = &mock.MooncakeBootstrapHandler{}
	mux := mock.NewMooncakePrefillMux(testInfo.prefillHandler, testInfo.bootstrapHandler)
	testInfo.prefillBackend = httptest.NewServer(mux)
	DeferCleanup(testInfo.prefillBackend.Close)

	// Set bootstrap port to match the test server's port
	prefillURL, err := url.Parse(testInfo.prefillBackend.URL)
	Expect(err).ToNot(HaveOccurred())
	mooncakeBootstrapPort, err = parsePort(prefillURL.Port())
	Expect(err).ToNot(HaveOccurred())

	// Proxy
	decodeURL, err := url.Parse(testInfo.decodeBackend.URL)
	Expect(err).ToNot(HaveOccurred())
	testInfo.decodeURL = decodeURL
	cfg := Config{Port: "0", DecoderURL: testInfo.decodeURL, KVConnector: KVConnectorMooncake}
	testInfo.proxy = NewProxy(cfg)

	return testInfo
}

func parsePort(portStr string) (int, error) {
	i, err := json.Number(portStr).Int64()
	if err != nil {
		return 0, err
	}
	return int(i), nil
}

var _ = Describe("Mooncake Connector", func() {

	var testInfo *mooncakeTestInfo

	BeforeEach(func() {
		testInfo = mooncakeTestSetup()
	})

	startProxy := func() string {
		go func() {
			defer GinkgoRecover()

			testInfo.proxy.allowlistValidator = &AllowlistValidator{enabled: false}
			err := testInfo.proxy.Start(testInfo.ctx)
			Expect(err).ToNot(HaveOccurred())

			testInfo.stoppedCh <- struct{}{}
		}()

		<-testInfo.proxy.readyCh
		DeferCleanup(func() {
			testInfo.cancelFn()
			<-testInfo.stoppedCh
		})

		return "http://" + testInfo.proxy.addr.String()
	}

	sendChatCompletionsRequest := func(proxyBaseAddr string) map[string]any {
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))

		var response map[string]any
		Expect(json.Unmarshal(responseBody, &response)).To(Succeed())
		return response
	}

	sendStreamingChatCompletionsRequest := func(proxyBaseAddr string) string {
		body := `{
				"model": "Qwen/Qwen2-0.5B",
				"messages": [
				  {"role": "user", "content": "Hello"}
				],
				"max_tokens": 50,
				"stream": true,
				"stream_options": {"include_usage": true}
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(body)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())
		defer rp.Body.Close()

		responseBody, err := io.ReadAll(rp.Body)
		Expect(err).ToNot(HaveOccurred())
		Expect(rp.StatusCode).To(Equal(http.StatusOK), string(responseBody))
		Expect(rp.Header.Get("Content-Type")).To(ContainSubstring(eventStreamContentType))
		return string(responseBody)
	}

	cachedTokensFromResponse := func(response map[string]any) float64 {
		usage, ok := response["usage"].(map[string]any)
		Expect(ok).To(BeTrue())
		details, ok := usage["prompt_tokens_details"].(map[string]any)
		Expect(ok).To(BeTrue())
		cachedTokens, ok := details["cached_tokens"].(float64)
		Expect(ok).To(BeTrue())
		return cachedTokens
	}

	It("should send prefill with transfer_id and construct decode kv_transfer_params", func() {
		proxyBaseAddr := startProxy()

		By("sending a /v1/chat/completions request with prefill header")
		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ChatCompletionsPath, bytes.NewReader([]byte(chatCompletionsRequestBody)))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:errcheck
			Fail(string(bp))
		}

		By("verifying prefill request has correct kv_transfer_params")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prefillReq := testInfo.prefillHandler.CompletionRequests[0]

		Expect(prefillReq).To(HaveKey(requestFieldKVTransferParams))
		kvTransferParams, ok := prefillReq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())

		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemoteDecode, true))
		Expect(kvTransferParams).To(HaveKeyWithValue(requestFieldDoRemotePrefill, false))
		Expect(kvTransferParams).To(HaveKey(requestFieldTransferID))
		transferID, ok := kvTransferParams[requestFieldTransferID].(string)
		Expect(ok).To(BeTrue())
		Expect(transferID).To(HavePrefix("xfer-"))

		// Mooncake prefill should NOT have remote_block_ids, remote_host, remote_port
		Expect(kvTransferParams).ToNot(HaveKey(requestFieldRemoteBlockIDs))
		Expect(kvTransferParams).ToNot(HaveKey(requestFieldRemoteHost))
		Expect(kvTransferParams).ToNot(HaveKey(requestFieldRemotePort))

		Expect(prefillReq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))
		Expect(prefillReq).To(HaveKeyWithValue("stream", false))
		Expect(prefillReq).ToNot(HaveKey("stream_options"))

		By("verifying decode request has sidecar-constructed kv_transfer_params")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]

		Expect(decodeReq).To(HaveKey(requestFieldKVTransferParams))
		decodeKVParams, ok := decodeReq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())

		Expect(decodeKVParams).To(HaveKeyWithValue(requestFieldDoRemotePrefill, true))
		Expect(decodeKVParams).To(HaveKeyWithValue(requestFieldDoRemoteDecode, false))
		Expect(decodeKVParams).To(HaveKeyWithValue(requestFieldRemoteEngineID, mock.TestMooncakeEngineID))
		Expect(decodeKVParams).To(HaveKey(requestFieldRemoteBootstrapAddr))
		Expect(decodeKVParams).To(HaveKeyWithValue(requestFieldTransferID, transferID))

		// Verify bootstrap addr format
		bootstrapAddr, ok := decodeKVParams[requestFieldRemoteBootstrapAddr].(string)
		Expect(ok).To(BeTrue())
		Expect(bootstrapAddr).To(HavePrefix("http://"))

		By("verifying bootstrap server was queried")
		Expect(testInfo.bootstrapHandler.QueryCount.Load()).To(BeNumerically("==", 1))
	})

	It("should propagate prefiller cached tokens to decode response", func() {
		proxyBaseAddr := startProxy()

		response := sendChatCompletionsRequest(proxyBaseAddr)

		Expect(cachedTokensFromResponse(response)).To(BeNumerically("==", 7))
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
	})

	It("should add prefiller cached tokens when decoder usage details omit cached_tokens", func() {
		testInfo.decodeHandler.RawResponse = `{"id":"chatcmpl-test","object":"chat.completion","choices":[],"usage":{"prompt_tokens":64,"completion_tokens":1,"total_tokens":65,"prompt_tokens_details":{}}}`
		proxyBaseAddr := startProxy()

		response := sendChatCompletionsRequest(proxyBaseAddr)

		Expect(cachedTokensFromResponse(response)).To(BeNumerically("==", 7))
	})

	It("should cache bootstrap engine_id and query only once per prefiller host", func() {
		proxyBaseAddr := startProxy()

		By("sending first request")
		sendChatCompletionsRequest(proxyBaseAddr)
		Expect(testInfo.bootstrapHandler.QueryCount.Load()).To(BeNumerically("==", 1))

		By("sending second request to same prefiller")
		sendChatCompletionsRequest(proxyBaseAddr)
		Expect(testInfo.bootstrapHandler.QueryCount.Load()).To(BeNumerically("==", 1))

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 2))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 2))
	})

	It("should replace cached tokens in streamed usage chunks", func() {
		testInfo.decodeHandler.RawResponseType = eventStreamContentType
		testInfo.decodeHandler.RawResponse = "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n\ndata: {\"choices\":[],\"usage\":{\"prompt_tokens\":64,\"completion_tokens\":1,\"total_tokens\":65,\"prompt_tokens_details\":{\"cached_tokens\":49}}}\n\ndata: [DONE]\n"
		proxyBaseAddr := startProxy()

		responseBody := sendStreamingChatCompletionsRequest(proxyBaseAddr)

		Expect(responseBody).To(ContainSubstring(`"content":"hello"`))
		Expect(responseBody).To(ContainSubstring(`"cached_tokens":7`))
		Expect(responseBody).ToNot(ContainSubstring(`"cached_tokens":49`))
		Expect(responseBody).To(ContainSubstring("data: [DONE]"))
	})

	It("should restore original max_tokens in decode request", func() {
		proxyBaseAddr := startProxy()

		sendChatCompletionsRequest(proxyBaseAddr)

		Expect(testInfo.prefillHandler.CompletionRequests).To(HaveLen(1))
		prefillReq := testInfo.prefillHandler.CompletionRequests[0]
		Expect(prefillReq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 1)))

		Expect(testInfo.decodeHandler.CompletionRequests).To(HaveLen(1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).To(HaveKeyWithValue("max_tokens", BeNumerically("==", 50)))
	})

	// Responses API tests

	It("should handle responses API request with mooncake connector", func() {
		proxyBaseAddr := startProxy()

		By("sending a /v1/responses request with prefill header")
		//nolint:goconst
		body := `{
				"model": "gpt-4o",
				"input": "Hello, how are you?",
				"max_output_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ResponsesPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())
		req.Header.Add(routing.PrefillEndpointHeader, testInfo.prefillBackend.URL[len("http://"):])

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		By("verifying prefill request has max_output_tokens=1")
		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		prefillReq := testInfo.prefillHandler.CompletionRequests[0]
		Expect(prefillReq).To(HaveKeyWithValue("max_output_tokens", BeNumerically("==", 1)))

		By("verifying decode request has original max_output_tokens=50")
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
		decodeReq := testInfo.decodeHandler.CompletionRequests[0]
		Expect(decodeReq).To(HaveKeyWithValue("max_output_tokens", BeNumerically("==", 50)))

		By("verifying decode has mooncake kv_transfer_params")
		decodeKVParams, ok := decodeReq[requestFieldKVTransferParams].(map[string]any)
		Expect(ok).To(BeTrue())
		Expect(decodeKVParams).To(HaveKeyWithValue(requestFieldRemoteEngineID, mock.TestMooncakeEngineID))
		Expect(decodeKVParams).To(HaveKey(requestFieldTransferID))
		Expect(decodeKVParams).To(HaveKey(requestFieldRemoteBootstrapAddr))
	})

	It("should pass through responses API request when no prefill header is set", func() {
		proxyBaseAddr := startProxy()

		body := `{
				"model": "gpt-4o",
				"input": "Hello, how are you?",
				"max_output_tokens": 50
			}`

		req, err := http.NewRequest(http.MethodPost, proxyBaseAddr+ResponsesPath, strings.NewReader(body))
		Expect(err).ToNot(HaveOccurred())

		rp, err := http.DefaultClient.Do(req)
		Expect(err).ToNot(HaveOccurred())

		if rp.StatusCode != 200 {
			bp, _ := io.ReadAll(rp.Body) //nolint:all
			Fail(string(bp))
		}

		Expect(testInfo.prefillHandler.RequestCount.Load()).To(BeNumerically("==", 0))
		Expect(testInfo.decodeHandler.RequestCount.Load()).To(BeNumerically("==", 1))
	})
})
