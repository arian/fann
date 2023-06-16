package vector_test

import (
	"testing"

	"github.com/arian/fann/vector"
)

func TestVectorAdd(t *testing.T) {
	t.Parallel()

	v1 := vector.Vector{1, 2, 3}
	v2 := vector.Vector{4, 5, 6}

	v3 := v1.Add(v2)
	if vector.Equal(v3, vector.Vector{5, 7, 9}) == false {
		t.Error("v3 should be {5, 7, 9}")
	}
	if vector.Equal(v1, v3) {
		t.Error("v1 and v3 should not be the same")
	}
}

func TestVectorSubtract(t *testing.T) {
	t.Parallel()

	v1 := vector.Vector{4, 5, 6}
	v2 := vector.Vector{1, 1, 2}

	v3 := v1.Subtract(v2)
	if vector.Equal(v3, vector.Vector{3, 4, 4}) == false {
		t.Error("v3 should be {3, 4, 4}")
	}
	if vector.Equal(v1, v3) {
		t.Error("v1 and v3 should not be the same")
	}
}

func TestVectorAvg(t *testing.T) {
	t.Parallel()

	v1 := vector.Vector{1, 2, 3}
	v2 := vector.Vector{4, 5, 6}

	v3 := v1.Avg(v2)
	if vector.Equal(v3, vector.Vector{2.5, 3.5, 4.5}) == false {
		t.Error("v1 should be {2.5, 3.5, 4.5}")
	}
	if vector.Equal(v1, v3) {
		t.Error("v1 and v3 should not be the same")
	}
}

func TestVectorDot(t *testing.T) {
	t.Parallel()

	v1 := vector.Vector{1, 2, 3}
	v2 := vector.Vector{4, 5, 6}

	dot := v1.Dot(v2)
	if dot != 32 {
		t.Errorf("dot should be 32, got %v", dot)
	}
}

func TestVectorSquareEucDistance(t *testing.T) {
	t.Parallel()

	v1 := vector.Vector{1, 2, 2}
	v2 := vector.Vector{4, 5, 6}

	d := v1.SquareEucDistance(v2)
	if d != 34 {
		t.Errorf("d should be 34, got %v", d)
	}
}


func TestVectorHash(t *testing.T) {
	t.Parallel()

	h1 := vector.Vector{1, 2, 2}.Hash()
	h2 := vector.Vector{4, 5, 6}.Hash()
	h3 := vector.Vector{4, 5, 6}.Hash()

	if h1 == h2 {
		t.Error("v1 and v2 should not have the same hash")
	}
	if h2 != h3 {
		t.Error("v2 and v3 should have the same hash")
	}
}
