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

package mock

import (
	"net/http"
	"sync/atomic"
)

const (
	// TestMooncakeEngineID is the engine_id returned by the mock bootstrap server.
	TestMooncakeEngineID = "test-mooncake-engine-id-550e8400"
)

// MooncakeBootstrapHandler serves the /query endpoint that the Mooncake connector
// uses to discover prefiller engine_ids.
type MooncakeBootstrapHandler struct {
	QueryCount atomic.Int32
}

func (h *MooncakeBootstrapHandler) ServeHTTP(w http.ResponseWriter, _ *http.Request) {
	h.QueryCount.Add(1)
	w.Header().Set("Content-Type", "application/json")
	w.Write([]byte(`{"0":{"engine_id":"` + TestMooncakeEngineID + `","worker_addr":{"0":{"0":"tcp://10.0.0.1:12345"}}}}`)) //nolint:all
}

// NewMooncakePrefillMux creates an http.ServeMux that routes /query to the
// bootstrap handler and all other paths to the chat completion handler.
func NewMooncakePrefillMux(completionHandler *ChatCompletionHandler, bootstrapHandler *MooncakeBootstrapHandler) *http.ServeMux {
	mux := http.NewServeMux()
	mux.Handle("/query", bootstrapHandler)
	mux.Handle("/", completionHandler)
	return mux
}
