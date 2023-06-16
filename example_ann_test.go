package main_test

import (
	"fmt"

	"github.com/arian/fann/fann"
	"github.com/arian/fann/vector"
)

func ExampleNewANNIndex() {

	vectors := []vector.Vector{
		{1, 2, 3},
		{3, 4, 5},
		{1, 2, 2},
		{1, 1, 2},
	}

	index := fann.NewANNIndex(
		3,
		2,
		vectors,
	)

	found := index.SearchApproximate(
		vector.Vector{1, 1, 1},
		2,
	)

	fmt.Println(found)

	for _, id := range found {
		fmt.Println(vectors[id])
	}

	// Output: [3 2]
	// [1 1 2]
	// [1 2 2]
}
