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

import (
	"testing"
)

func Test_cblas_cdotc_sub(t *testing.T) {
	type args struct {
		N    int
		X    []complex64
		incX int
		Y    []complex64
		incY int
	}
	tests := []struct {
		name    string
		args    args
		wantRet complex64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex64{1 + 1i, 1 + 1i},
				incX: 1,
				Y:    []complex64{1 + 1i, 1 + 1i},
				incY: 1,
			},
			wantRet: 4 + 0i,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := cblas_cdotc_sub(tt.args.N, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); gotRet != tt.wantRet {
				t.Errorf("cblas_cdotc_sub() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func Test_cblas_cdotu_sub(t *testing.T) {
	type args struct {
		N    int
		X    []complex64
		incX int
		Y    []complex64
		incY int
	}
	tests := []struct {
		name    string
		args    args
		wantRet complex64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex64{1 + 1i, 1 + 1i},
				incX: 1,
				Y:    []complex64{1 + 1i, 1 + 1i},
				incY: 1,
			},
			wantRet: 0 + 4i,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := cblas_cdotu_sub(tt.args.N, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); gotRet != tt.wantRet {
				t.Errorf("cblas_cdotu_sub() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func Test_cblas_dasum(t *testing.T) {
	type args struct {
		N    int
		X    []float64
		incX int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []float64{1, 2},
				incX: 1,
			},
			want: 3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_dasum(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_dasum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_ddot(t *testing.T) {
	type args struct {
		N    int
		X    []float64
		incX int
		Y    []float64
		incY int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []float64{1, 2},
				incX: 1,
				Y:    []float64{1, 2},
				incY: 1,
			},
			want: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_ddot(tt.args.N, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); got != tt.want {
				t.Errorf("cblas_ddot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_dnrm2(t *testing.T) {
	type args struct {
		N    int
		X    []float64
		incX int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []float64{2, 0},
				incX: 1,
			},
			want: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_dnrm2(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_dnrm2() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_dsdot(t *testing.T) {
	type args struct {
		N    int
		X    []float32
		incX int
		Y    []float32
		incY int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []float32{1, 2},
				incX: 1,
				Y:    []float32{1, 2},
				incY: 1,
			},
			want: 5.0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_dsdot(tt.args.N, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); got != tt.want {
				t.Errorf("cblas_dsdot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_dzasum(t *testing.T) {
	type args struct {
		N    int
		X    []complex128
		incX int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex128{2 + 1i, 1 + 1i},
				incX: 1,
			},
			want: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_dzasum(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_dzasum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_dznrm2(t *testing.T) {
	type args struct {
		N    int
		X    []complex128
		incX int
	}
	tests := []struct {
		name string
		args args
		want float64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex128{0 + 2i, 0 + 0i},
				incX: 1,
			},
			want: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_dznrm2(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_dznrm2() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_sasum(t *testing.T) {
	type args struct {
		N    int
		X    []float32
		incX int
	}
	tests := []struct {
		name string
		args args
		want float32
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []float32{1, 2},
				incX: 1,
			},
			want: 3,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_sasum(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_sasum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_scasum(t *testing.T) {
	type args struct {
		N    int
		X    []complex64
		incX int
	}
	tests := []struct {
		name string
		args args
		want float32
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex64{2 + 0i, 3 + 0i},
				incX: 1,
			},
			want: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_scasum(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_scasum() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_scnrm2(t *testing.T) {
	type args struct {
		N    int
		X    []complex64
		incX int
	}
	tests := []struct {
		name string
		args args
		want float32
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex64{0 + 0i, 2 + 0i},
				incX: 1,
			},
			want: 2,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_scnrm2(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_scnrm2() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_sdot(t *testing.T) {
	type args struct {
		N    int
		X    []float32
		incX int
		Y    []float32
		incY int
	}
	tests := []struct {
		name string
		args args
		want float32
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []float32{1, 2},
				incX: 1,
				Y:    []float32{1, 2},
				incY: 1,
			},
			want: 5,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_sdot(tt.args.N, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); got != tt.want {
				t.Errorf("cblas_sdot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_sdsdot(t *testing.T) {
	type args struct {
		N     int
		alpha float32
		X     []float32
		incX  int
		Y     []float32
		incY  int
	}
	tests := []struct {
		name string
		args args
		want float32
	}{
		{
			name: "1",
			args: args{
				N:     2,
				alpha: 1,
				X:     []float32{1, 2},
				incX:  1,
				Y:     []float32{2, 2},
				incY:  1,
			},
			want: 7,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_sdsdot(tt.args.N, tt.args.alpha, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); got != tt.want {
				t.Errorf("cblas_sdsdot() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_snrm2(t *testing.T) {
	type args struct {
		N    int
		X    []float32
		incX int
	}
	tests := []struct {
		name string
		args args
		want float32
	}{
		{
			name: "1",
			args: args{
				N:    3,
				X:    []float32{1, 2, 4},
				incX: 1,
			},
			want: 4.52576,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := cblas_snrm2(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
				t.Errorf("cblas_snrm2() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_cblas_zdotc_sub(t *testing.T) {
	type args struct {
		N    int
		X    []complex64
		incX int
		Y    []complex64
		incY int
	}
	tests := []struct {
		name    string
		args    args
		wantRet complex64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex64{1, 2},
				incX: 1,
				Y:    []complex64{1, 2},
				incY: 1,
			},
			wantRet: 0 + 0i,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := cblas_zdotc_sub(tt.args.N, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); gotRet != tt.wantRet {
				t.Errorf("cblas_zdotc_sub() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}

func Test_cblas_zdotu_sub(t *testing.T) {
	type args struct {
		N    int
		X    []complex64
		incX int
		Y    []complex64
		incY int
	}
	tests := []struct {
		name    string
		args    args
		wantRet complex64
	}{
		{
			name: "1",
			args: args{
				N:    2,
				X:    []complex64{1, 2},
				incX: 1,
				Y:    []complex64{1, 2},
				incY: 1,
			},
			wantRet: 0,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if gotRet := cblas_zdotu_sub(tt.args.N, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY); gotRet != tt.wantRet {
				t.Errorf("cblas_zdotu_sub() = %v, want %v", gotRet, tt.wantRet)
			}
		})
	}
}
