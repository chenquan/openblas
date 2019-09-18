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
	"fmt"
	"testing"
)

func Test_cblasSdsdot(t *testing.T) {
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
	}{
		{name: "1",
			args: args{
				N:     1,
				alpha: 4343,
				X:     []float32{1, 2},
				incX:  1,
				Y:     []float32{4, 2},
				incY:  1,
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sdsdot := cblas_sdsdot(tt.args.N, tt.args.alpha, tt.args.X, tt.args.incX, tt.args.Y, tt.args.incY)
			fmt.Println(sdsdot)
		})
	}
}

func Test_cblas_ddot(t *testing.T) {
	ddot := cblas_ddot(2, []float64{9, 4}, 1, []float64{4, 2}, 1)
	fmt.Println(ddot)
}

func Test_cblas_dsdot(t *testing.T) {
	result := cblas_dsdot(2, []float32{9, 4}, 1, []float32{4, 2}, 1)
	fmt.Println(result)
}

func Test_cblas_sdot(t *testing.T) {
	result := cblas_sdot(2, []float32{9, 4}, 1, []float32{4, 2}, 1)
	fmt.Println(result)

}

func Test_cblas_sdsdot(t *testing.T) {
	result := cblas_sdsdot(2, 3, []float32{1, 2}, 1, []float32{1, 2}, 1)
	fmt.Println(result)
}

func Test_cblas_ssum(t *testing.T) {
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
			//if got := cblas_ssum(tt.args.N, tt.args.X, tt.args.incX); got != tt.want {
			//	t.Errorf("cblas_ssum() = %v, want %v", got, tt.want)
			//}
		})
	}
}
