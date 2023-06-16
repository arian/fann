package fann

import (
	"math/rand"
	"sort"

	"github.com/arian/fann/vector"
)

type hyperplane struct {
	normal vector.Vector
	d      float64
}

type node struct {
	inner *innerNode
	leaf  []int
}

type innerNode struct {
	hyperplane *hyperplane
	above      *node
	below      *node
}

type idWithDistance struct {
	id       int
	distance float64
}

type ANNIndex struct {
	roots  []*node
	values map[int]vector.Vector
	dups   map[int][]int
}

func NewANNIndex(
	numTrees int,
	maxSize int,
	points []vector.Vector,
) *ANNIndex {

	uniqueVectors, dups := uniqueVectors(points)
	indexes := make([]int, 0, len(uniqueVectors))
	for i := range uniqueVectors {
		indexes = append(indexes, i)
	}

	roots := make([]*node, 0, numTrees)
	for i := 0; i < numTrees; i++ {
		rand := rand.New(rand.NewSource(int64(i)))
		roots = append(roots, buildNode(rand, maxSize, indexes, uniqueVectors))
	}

	return &ANNIndex{
		roots:  roots,
		values: uniqueVectors,
		dups:   dups,
	}
}

func buildHyperplane(
	rand *rand.Rand,
	indexes []int,
	allVectors map[int]vector.Vector,
) (*hyperplane, []int, []int) {

	l := len(indexes)
	var a_i, b_i int
	for {
		a_i = rand.Intn(l)
		b_i = rand.Intn(l)
		if a_i != b_i {
			break
		}
	}

	a := allVectors[indexes[a_i]]
	b := allVectors[indexes[b_i]]

	// cartesian eq for hyperplane n * (x - x_0) = 0
	// n (normal vector) is the coefs x_1 to x_n
	coefficients := a.Subtract(b)
	pointOnPlane := a.Avg(b)
	constant := coefficients.Dot(pointOnPlane) * -1.0

	hyperplane := &hyperplane{
		normal: coefficients,
		d:      constant,
	}

	var above, below []int

	for _, idx := range indexes {
		point := allVectors[idx]
		if hyperplane.IsAbove(point) {
			above = append(above, idx)
		} else {
			below = append(below, idx)
		}
	}

	return hyperplane, above, below
}

func buildNode(
	rand *rand.Rand,
	maxSize int,
	indexes []int,
	vectors map[int]vector.Vector,
) *node {
	if len(indexes) <= maxSize {
		return &node{leaf: indexes}
	}

	plane, above, below := buildHyperplane(rand, indexes, vectors)

	nodeAbove := buildNode(rand, maxSize, above, vectors)
	nodeBelow := buildNode(rand, maxSize, below, vectors)

	return &node{
		inner: &innerNode{
			hyperplane: plane,
			above:      nodeAbove,
			below:      nodeBelow,
		},
	}
}

func (i *ANNIndex) treeResult(
	query vector.Vector,
	n int,
	tree *node,
	candidates map[int]struct{},
) int {
	if tree.inner != nil {

		var main, backup *node

		if tree.inner.hyperplane.IsAbove(query) {
			main = tree.inner.above
			backup = tree.inner.below
		} else {
			main = tree.inner.below
			backup = tree.inner.above
		}

		k := i.treeResult(query, n, main, candidates)

		if k >= n {
			return k
		}

		return k + i.treeResult(query, n-k, backup, candidates)

	} else {
		k := len(tree.leaf)
		// if n < k {
		// k = n
		// }
		for i := 0; i < k; i++ {
			candidates[tree.leaf[i]] = struct{}{}
		}
		return k
	}
}

func (i *ANNIndex) SearchApproximate(
	point vector.Vector,
	k int,
) []int {

	candidates := make(map[int]struct{})

	for _, root := range i.roots {
		i.treeResult(point, k, root, candidates)
	}

	var distances []idWithDistance

	for id := range candidates {
		v := i.values[id]
		d := v.SquareEucDistance(point)
		distances = append(distances, idWithDistance{id, d})
		for _, dup := range i.dups[id] {
			distances = append(distances, idWithDistance{dup, d})
		}
	}

	sort.Slice(distances, func(i, j int) bool {
		return distances[i].distance < distances[j].distance
	})

	if len(distances) > k {
		distances = distances[:k]
	}

	ids := make([]int, len(distances))
	for i, v := range distances {
		ids[i] = v.id
	}

	return ids
}

func uniqueVectors(vectors []vector.Vector) (map[int]vector.Vector, map[int][]int) {
	unique := make(map[int]vector.Vector)
	seen := make(map[uint64][]int)
	for i, v := range vectors {
		hash := v.Hash()
		if len(seen[hash]) == 0 {
			unique[i] = v
		}
		seen[hash] = append(seen[hash], i)
	}
	dups := make(map[int][]int)
	for _, ids := range seen {
		if len(ids) > 1 {
			dups[ids[0]] = ids[1:]
		}
	}
	return unique, dups
}

func (h hyperplane) IsAbove(point vector.Vector) bool {
	return h.normal.Dot(point)+h.d > 0
}
