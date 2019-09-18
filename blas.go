//   Copyright 2019 YunQi
//
//   Licensed under the Apache License, Version 2.0 (the "License");
//   you may not use this file except in compliance with the License.
//   You may obtain a copy of the License at
//
//       http://www.apache.org/licenses/LICENSE-2.0
//
//   Unless required by applicable law or agreed to in writing, software
//   distributed under the License is distributed on an "AS IS" BASIS,
//   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//   See the License for the specific language governing permissions and
//   limitations under the License.
package openblas

/*
#cgo CFLAGS: -I./blas/include
#cgo LDFLAGS: -L./blas/lib -lopenblas
#include"./blas/include/cblas.h"
*/
import "C"
import (
	"fmt"
	"unsafe"
)

func cblas_sdsdot(N int, alpha float32, X []float32, incX int, Y []float32, incY int) float32 {
	var result C.float = C.cblas_sdsdot(C.int(N), C.float(alpha), (*C.float)(&X[0]), C.int(incX), (*C.float)(&Y[0]), C.int(incY))
	return float32(result)
}

func cblas_dsdot(N int, X []float32, incX int, Y []float32, incY int) float64 {
	var result C.double = C.cblas_dsdot(C.int(N), (*C.float)(&X[0]), C.int(incX), (*C.float)(&Y[0]), C.int(incY))
	return float64(result)
}
func cblas_sdot(N int, X []float32, incX int, Y []float32, incY int) float32 {
	var result C.float = C.cblas_sdot(C.int(N), (*C.float)(&X[0]), C.int(incX), (*C.float)(&Y[0]), C.int(incY))
	return float32(result)
}
func cblas_ddot(N int, X []float64, incX int, Y []float64, incY int) float64 {
	var result C.double = C.cblas_ddot(C.int(N), (*C.double)(&X[0]), C.int(incX), (*C.double)(&Y[0]), C.int(incY))
	fmt.Println(result)
	return float64(result)
}

//openblas_complex_float  cblas_cdotu(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy);
//openblas_complex_float  cblas_cdotc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void  *y, OPENBLAS_CONST blasint incy);
//openblas_complex_double cblas_zdotu(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy);
//openblas_complex_double cblas_zdotc(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx, OPENBLAS_CONST void *y, OPENBLAS_CONST blasint incy);

func cblas_cdotu_sub(N int, X []complex64, incX int, Y []complex64, incY int) (ret complex64) {
	C.cblas_cdotu_sub(C.int(N), unsafe.Pointer(&X[0]), C.int(incX), unsafe.Pointer(&Y[0]), C.int(incY), unsafe.Pointer(&ret))
	return ret
}
func cblas_cdotc_sub(N int, X []complex64, incX int, Y []complex64, incY int) (ret complex64) {
	C.cblas_cdotc_sub(C.int(N), unsafe.Pointer(&X[0]), C.int(incX), unsafe.Pointer(&Y[0]), C.int(incY), unsafe.Pointer(&ret))
	return ret
}
func cblas_zdotu_sub(N int, X []complex64, incX int, Y []complex64, incY int) (ret complex64) {
	C.cblas_zdotu_sub(C.int(N), unsafe.Pointer(&X[0]), C.int(incX), unsafe.Pointer(&Y[0]), C.int(incY), unsafe.Pointer(&ret))
	return ret
}
func cblas_zdotc_sub(N int, X []complex64, incX int, Y []complex64, incY int) (ret complex64) {
	C.cblas_zdotc_sub(C.int(N), unsafe.Pointer(&X[0]), C.int(incX), unsafe.Pointer(&Y[0]), C.int(incY), unsafe.Pointer(&ret))
	return ret
}

func cblas_sasum(N int, X []float32, incX int) float32 {
	return float32(C.cblas_sasum(C.int(N), (*C.float)(&X[0]), C.int(incX)))
}

func cblas_dasum(N int, X []float64, incX int) float64 {
	return float64(C.cblas_dasum(C.int(N), (*C.double)(&X[0]), C.int(incX)))
}
func cblas_scasum(N int, X []complex64, incX int) float32 {
	return float32(C.cblas_scasum(C.int(N), unsafe.Pointer(&X[0]), C.int(incX)))
}
func cblas_dzasum(N int, X []complex128, incX int) float64 {
	return float64(C.cblas_dzasum(C.int(N), unsafe.Pointer(&X[0]), C.int(incX)))
}

//func cblas_ssum(N int, X []float32, incX int) float32 {
//	return float32(C.cblas_ssum(C.int(N), (*C.float)(&X[0]), C.int(incX)))
//}
//func cblas_dsum(N int, X []float64, incX int) float64 {
//	return float64(C.cblas_dsum(C.int(N), (*C.double)(&X[0]), C.int(incX)))
//}
//func cblas_scsum(N int, X []complex64, incX int) float32 {
//	return float32(C.cblas_scsum(C.int(N), unsafe.Pointer(&X[0]), C.int(incX)))
//}
//func cblas_dzsum(N int, X []complex128, incX int) float64 {
//	return float64(C.cblas_dzsum(C.int(N), unsafe.Pointer(&X[0]), C.int(incX)))
//}

func cblas_snrm2(N int, X []float32, incX int) float32 {
	return float32(C.cblas_snrm2(C.int(N), (*C.float)(&X[0]), C.int(incX)))
}
func cblas_dnrm2(N int, X []float64, incX int) float64 {
	return float64(C.cblas_dnrm2(C.int(N), (*C.double)(&X[0]), C.int(incX)))
}
func cblas_scnrm2(N int, X []complex64, incX int) float32 {
	return float32(C.cblas_scnrm2(C.int(N), unsafe.Pointer(&X[0]), C.int(incX)))

}
func cblas_dznrm2(N int, X []complex128, incX int) float64 {
	return float64(C.cblas_dznrm2(C.int(N), unsafe.Pointer(&X[0]), C.int(incX)))
}

//CBLAS_INDEX cblas_isamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_idamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_icamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_izamax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

//CBLAS_INDEX cblas_isamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_idamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_icamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_izamin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

//CBLAS_INDEX cblas_ismax(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_idmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_icmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_izmax(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);

//CBLAS_INDEX cblas_ismin(OPENBLAS_CONST blasint n, OPENBLAS_CONST float  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_idmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST double *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_icmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void  *x, OPENBLAS_CONST blasint incx);
//CBLAS_INDEX cblas_izmin(OPENBLAS_CONST blasint n, OPENBLAS_CONST void *x, OPENBLAS_CONST blasint incx);
