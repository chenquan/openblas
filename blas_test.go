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
