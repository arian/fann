package fann_test

import (
	"testing"

	"github.com/arian/fann/fann"
	"github.com/arian/fann/vector"
)

func TestAnnIndex(t *testing.T) {

	idx := fann.NewANNIndex(
		3,
		2,
		[]vector.Vector{
			{1, 2, 3},
			{3, 4, 5},
		},
	)

	found := idx.SearchApproximate(
		vector.Vector{1, 1, 2},
		2,
	)

	expectedIds := []int{0, 1}
	if len(found) != len(expectedIds) {
		t.Fatalf("expected %d results, got %d", len(expectedIds), len(found))
	}
	for i, id := range found {
		if id != expectedIds[i] {
			t.Errorf("expected vector %d, got %v", expectedIds[i], id)
		}
	}
}

func TestAnnIndexBranching(t *testing.T) {

	idx := fann.NewANNIndex(
		3,
		2,
		[]vector.Vector{
			{1, 2, 3},
			{3, 4, 5},
			{1, 2, 2},
		},
	)

	found := idx.SearchApproximate(
		vector.Vector{1, 1, 2},
		2,
	)

	if len(found) != 2 {
		t.Fatalf("expected 2 results, got %d", len(found))
	}
	if found[0] != 2 {
		t.Errorf("expected vector 2, got %v", found[0])
	}
	if found[1] != 0 {
		t.Errorf("expected vector 0, got %v", found[1])
	}
}

func TestDuplicatesVectors(t *testing.T) {

	idx := fann.NewANNIndex(
		3,
		2,
		[]vector.Vector{
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 3},
			{1, 2, 2},
		},
	)

	found := idx.SearchApproximate(vector.Vector{1, 1, 2}, 3)

	if len(found) != 3 {
		t.Fatalf("expected 3 results, got %d", len(found))
	}
	expectedIds := []int{3, 0, 1}
	for i, id := range found {
		if id != expectedIds[i] {
			t.Errorf("expected vector %d, got %v", expectedIds[i], id)
		}
	}
}
