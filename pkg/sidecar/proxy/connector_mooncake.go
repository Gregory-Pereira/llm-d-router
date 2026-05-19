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
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"os"
	"strconv"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/llm-d/llm-d-router/pkg/telemetry"
)

var mooncakeBootstrapPort int

func init() {
	mooncakeBootstrapPort = 8998

	if portStr := os.Getenv("VLLM_MOONCAKE_BOOTSTRAP_PORT"); portStr != "" {
		if port, err := strconv.Atoi(portStr); err == nil {
			mooncakeBootstrapPort = port
		}
	}
}

func (s *Server) runMooncakeProtocol(w http.ResponseWriter, r *http.Request, prefillPodHostPort string, apiType APIType) {
	tokenLimitFields := tokenLimitFieldsForAPIType(apiType)
	s.logger.V(4).Info("running Mooncake protocol", "url", prefillPodHostPort, "tokenLimitFields", tokenLimitFields)

	defer r.Body.Close() //nolint:errcheck
	original, err := io.ReadAll(r.Body)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		w.Write([]byte(err.Error())) //nolint:errcheck
		return
	}

	var completionRequest map[string]any
	if err := json.Unmarshal(original, &completionRequest); err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	requestUUID, err := uuid.NewUUID()
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	uuidStr := requestUUID.String()
	transferID := "xfer-" + uuidStr

	// Prefill Stage
	tracer := telemetry.Tracer()
	ctx := r.Context()

	ctx, prefillSpan := tracer.Start(ctx, "llm_d.pd_proxy.prefill",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	prefillSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.request_id", uuidStr),
		attribute.String("llm_d.pd_proxy.prefill_target", prefillPodHostPort),
		attribute.String("llm_d.pd_proxy.connector", "mooncake"),
	)
	prefillStart := time.Now()

	preq := r.Clone(ctx)
	preq.Header.Add(requestHeaderRequestID, uuidStr)

	streamValue, streamOk := completionRequest[requestFieldStream]
	streamOptionsValue, streamOptionsOk := completionRequest[requestFieldStreamOptions]

	type savedField struct {
		field   string
		val     any
		present bool
	}
	var savedTokenValues [2]savedField
	for i, field := range tokenLimitFields {
		if v, ok := completionRequest[field]; ok {
			savedTokenValues[i] = savedField{field: field, val: v, present: true}
		} else {
			savedTokenValues[i] = savedField{field: field}
		}
	}

	originalRequest := maps.Clone(completionRequest)

	completionRequest[requestFieldKVTransferParams] = map[string]any{
		requestFieldDoRemoteDecode:  true,
		requestFieldDoRemotePrefill: false,
		requestFieldTransferID:      transferID,
	}

	completionRequest[requestFieldStream] = false
	delete(completionRequest, requestFieldStreamOptions)

	for _, field := range tokenLimitFields {
		completionRequest[field] = 1
	}

	pbody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	preq.Body = io.NopCloser(bytes.NewReader(pbody))
	preq.ContentLength = int64(len(pbody))

	prefillHandler, err := s.prefillerProxyHandler(prefillPodHostPort)
	if err != nil {
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	s.logger.V(4).Info("sending prefill request", "to", prefillPodHostPort)
	s.logger.V(5).Info("Prefill request", "body", string(pbody))
	pw := &bufferedResponseWriter{}
	prefillHandler.ServeHTTP(pw, preq)

	prefillDuration := time.Since(prefillStart)
	prefillSpan.SetAttributes(
		attribute.Int("llm_d.pd_proxy.prefill.status_code", pw.statusCode),
		attribute.Float64("llm_d.pd_proxy.prefill.duration_ms", float64(prefillDuration.Milliseconds())),
	)

	if isHTTPError(pw.statusCode) {
		s.logger.Error(err, "request failed", "code", pw.statusCode, "body", pw.buffer.String())
		prefillSpan.SetStatus(codes.Error, "prefill request failed")
		prefillSpan.End()

		if shouldFallbackToDecode(pw) {
			s.logger.Info("fallback to decode", "request_id", uuidStr)
			fallbackReq := cloneRequestWithBody(r, original)
			s.dispatchDecode(w, fallbackReq, originalRequest)
		} else {
			for key, values := range pw.Header() {
				for _, v := range values {
					w.Header().Add(key, v)
				}
			}
			w.WriteHeader(pw.statusCode)
			_, err := w.Write(pw.bodyBytes())
			if err != nil {
				s.logger.Error(err, "failed to send error response to client")
			}
		}
		return
	}
	prefillSpan.End()

	var prefillerResponse map[string]any
	if err := json.Unmarshal(pw.bodyBytes(), &prefillerResponse); err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	pCachedTokens, hasPCachedTokens := extractCachedTokens(prefillerResponse)
	if !hasPCachedTokens {
		pCachedTokens = 0
	}

	// Bootstrap discovery: get engine_id for this prefiller
	prefillHost := s.getBootstrapHost(prefillPodHostPort)
	bootstrapAddr := fmt.Sprintf("http://%s:%d", prefillHost, mooncakeBootstrapPort)

	engineID, err := s.getMooncakeEngineID(prefillHost, bootstrapAddr)
	if err != nil {
		s.logger.Error(err, "failed to discover mooncake engine_id", "bootstrapAddr", bootstrapAddr)
		if err := errorBadGateway(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}

	s.logger.V(5).Info("mooncake bootstrap discovery complete",
		"engineID", engineID,
		"bootstrapAddr", bootstrapAddr,
		"transferID", transferID,
	)

	// Decode Stage
	ctx, decodeSpan := tracer.Start(ctx, "llm_d.pd_proxy.decode",
		trace.WithSpanKind(trace.SpanKindInternal),
	)
	defer decodeSpan.End()

	decodeSpan.SetAttributes(
		attribute.String("llm_d.pd_proxy.request_id", uuidStr),
		attribute.String("llm_d.pd_proxy.connector", "mooncake"),
	)
	decodeStart := time.Now()

	dreq := r.Clone(ctx)
	dreq.Header.Add(requestHeaderRequestID, uuidStr)

	delete(completionRequest, requestFieldStream)
	streamingEnabled := false
	if streamOk {
		completionRequest[requestFieldStream] = streamValue
		if streamBool, ok := streamValue.(bool); ok {
			streamingEnabled = streamBool
		}
	}
	decodeSpan.SetAttributes(attribute.Bool("llm_d.pd_proxy.decode.streaming", streamingEnabled))
	if streamOptionsOk {
		completionRequest[requestFieldStreamOptions] = streamOptionsValue
	}

	for i := range savedTokenValues[:len(tokenLimitFields)] {
		sv := &savedTokenValues[i]
		delete(completionRequest, sv.field)
		if sv.present {
			completionRequest[sv.field] = sv.val
		}
	}

	// Construct kv_transfer_params for decode (Mooncake does NOT pass through from prefill)
	completionRequest[requestFieldKVTransferParams] = map[string]any{
		requestFieldDoRemotePrefill:     true,
		requestFieldDoRemoteDecode:      false,
		requestFieldRemoteBootstrapAddr: bootstrapAddr,
		requestFieldRemoteEngineID:      engineID,
		requestFieldTransferID:          transferID,
	}

	dbody, err := json.Marshal(completionRequest)
	if err != nil {
		if err := errorJSONInvalid(err, w); err != nil {
			s.logger.Error(err, "failed to send error response to client")
		}
		return
	}
	dreq.Body = io.NopCloser(bytes.NewReader(dbody))
	dreq.ContentLength = int64(len(dbody))

	s.logger.V(5).Info("sending request to decoder", "body", string(dbody))
	decodeWriter, finalizeDecodeWriter := newCachedTokensResponseWriterWithFinalize(w, pCachedTokens)
	dataParallelUsed := s.forwardDataParallel && s.dataParallelHandler(decodeWriter, dreq)
	decodeSpan.SetAttributes(attribute.Bool("llm_d.pd_proxy.decode.data_parallel", dataParallelUsed))

	if !dataParallelUsed {
		s.logger.V(4).Info("sending request to decoder", "to", s.config.DecoderURL.Host)
		decodeSpan.SetAttributes(attribute.String("llm_d.pd_proxy.decode.target", s.config.DecoderURL.Host))
		s.dispatchDecode(decodeWriter, dreq, completionRequest)
	}
	if err := finalizeDecodeWriter(); err != nil {
		s.logger.Error(err, "failed to flush cached token response writer")
		decodeSpan.SetStatus(codes.Error, "failed to flush cached token response writer")
		return
	}

	decodeDuration := time.Since(decodeStart)
	decodeSpan.SetAttributes(attribute.Float64("llm_d.pd_proxy.decode.duration_ms", float64(decodeDuration.Milliseconds())))

	if currentSpan := trace.SpanFromContext(ctx); currentSpan.SpanContext().IsValid() {
		var totalDuration time.Duration
		var trueTTFT time.Duration
		if requestStartValue := ctx.Value(requestStartTimeKey); requestStartValue != nil {
			if requestStart, ok := requestStartValue.(time.Time); ok {
				totalDuration = time.Since(requestStart)
				trueTTFT = decodeStart.Sub(requestStart)
			}
		}

		coordinatorOverhead := decodeStart.Sub(prefillStart.Add(prefillDuration))

		currentSpan.SetAttributes(
			attribute.Float64("llm_d.pd_proxy.total_duration_ms", float64(totalDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.true_ttft_ms", float64(trueTTFT.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.prefill_duration_ms", float64(prefillDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.decode_duration_ms", float64(decodeDuration.Milliseconds())),
			attribute.Float64("llm_d.pd_proxy.coordinator_overhead_ms", float64(coordinatorOverhead.Milliseconds())),
		)
	}
}

// getMooncakeEngineID returns the engine_id for the given prefiller host,
// querying the bootstrap server if not already cached.
func (s *Server) getMooncakeEngineID(prefillHost, bootstrapAddr string) (string, error) {
	if engineID, ok := s.mooncakeBootstrapCache.Get(prefillHost); ok {
		return engineID, nil
	}

	queryURL := bootstrapAddr + "/query"
	s.logger.V(4).Info("querying mooncake bootstrap server", "url", queryURL)

	resp, err := http.Get(queryURL) //nolint:gosec
	if err != nil {
		return "", fmt.Errorf("failed to query mooncake bootstrap server at %s: %w", queryURL, err)
	}
	defer resp.Body.Close() //nolint:errcheck

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("mooncake bootstrap server returned status %d", resp.StatusCode)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("failed to read bootstrap response: %w", err)
	}

	// Response format: {"0": {"engine_id": "...", "worker_addr": {...}}, ...}
	var bootstrapResponse map[string]map[string]any
	if err := json.Unmarshal(body, &bootstrapResponse); err != nil {
		return "", fmt.Errorf("failed to parse bootstrap response: %w", err)
	}

	for _, dpEntry := range bootstrapResponse {
		engineID, ok := dpEntry["engine_id"].(string)
		if !ok || engineID == "" {
			continue
		}
		s.mooncakeBootstrapCache.Add(prefillHost, engineID)
		return engineID, nil
	}

	return "", fmt.Errorf("no engine_id found in bootstrap response from %s", queryURL)
}
