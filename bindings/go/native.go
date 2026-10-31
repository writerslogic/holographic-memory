// Copyright 2024-2026 WritersLogic Contributors
// SPDX-License-Identifier: Apache-2.0

package warm

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo darwin,arm64 LDFLAGS: -L${SRCDIR}/lib/darwin_arm64
#cgo darwin,amd64 LDFLAGS: -L${SRCDIR}/lib/darwin_amd64
#cgo linux,arm64 LDFLAGS: -L${SRCDIR}/lib/linux_arm64
#cgo linux,amd64 LDFLAGS: -L${SRCDIR}/lib/linux_amd64
#cgo windows,arm64 LDFLAGS: -L${SRCDIR}/lib/windows_arm64
#cgo windows,amd64 LDFLAGS: -L${SRCDIR}/lib/windows_amd64
#cgo LDFLAGS: -lwriterslogic_warm_c
#cgo darwin LDFLAGS: -framework CoreFoundation -framework Security -liconv
#cgo linux LDFLAGS: -ldl -lpthread -lm
#cgo windows LDFLAGS: -lws2_32 -luserenv -lbcrypt -lntdll -lkernel32 -ladvapi32

#include <stdlib.h>
#include "warm_go.h"
*/
import "C"

import (
	"runtime"
	"unsafe"
)

// pins keeps Go memory referenced from C-allocated structs alive and unmoved for one call.
type pins struct {
	pinner runtime.Pinner
	c      []unsafe.Pointer
}

func (p *pins) release() {
	p.pinner.Unpin()
	for _, ptr := range p.c {
		C.free(ptr)
	}
	p.c = nil
}

// calloc returns zeroed C memory freed by release.
func (p *pins) calloc(ptr unsafe.Pointer) unsafe.Pointer {
	if ptr == nil {
		panic("hms: C allocation failed")
	}
	p.c = append(p.c, ptr)
	return ptr
}

// str passes a Go string to C without copying: its bytes are pinned for the call.
func (p *pins) str(s string, present bool) C.warm_str_t {
	if !present {
		return C.warm_go_str(nil, 0, 0)
	}
	data := unsafe.StringData(s)
	if len(s) > 0 {
		p.pinner.Pin(data)
	}
	return C.warm_go_str((*C.uint8_t)(unsafe.Pointer(data)), C.size_t(len(s)), 1)
}

func (p *pins) floats(values []float64) *C.double {
	if len(values) == 0 {
		return nil
	}
	p.pinner.Pin(&values[0])
	return (*C.double)(unsafe.Pointer(&values[0]))
}

func goStr(s C.warm_str_t) string {
	if s.ptr == nil || s.len == 0 {
		return ""
	}
	return C.GoStringN((*C.char)(unsafe.Pointer(s.ptr)), C.int(s.len))
}

func lastError(code C.warm_status_t) error {
	if code == C.WARM_OK {
		return nil
	}
	return &Error{Code: Code(code), Message: C.GoString(C.hms1_last_error_message())}
}
