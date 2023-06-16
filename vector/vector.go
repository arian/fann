package vector

import (
	"encoding/binary"
	"hash"
	"hash/fnv"
)

type Vector []float64

func Equal(v1, v2 Vector) bool {
	if len(v1) != len(v2) {
		return false
	}
	for i, v := range v1 {
		if v != v2[i] {
			return false
		}
	}
	return true
}

func (v Vector) Add(v2 Vector) Vector {
	l := len(v)
	if len(v2) < l {
		l = len(v2)
	}
	v3 := make(Vector, l)
	for i := 0; i < l; i++ {
		v3[i] = v[i] + v2[i]
	}
	return v3
}

func (v Vector) Subtract(v2 Vector) Vector {
	l := len(v)
	if len(v2) < l {
		l = len(v2)
	}
	v3 := make(Vector, l)
	for i := 0; i < l; i++ {
		v3[i] = v[i] - v2[i]
	}
	return v3
}

func (v Vector) Avg(v2 Vector) Vector {
	l := len(v)
	if len(v2) < l {
		l = len(v2)
	}
	v3 := make(Vector, l)
	for i := 0; i < l; i++ {
		v3[i] = (v[i] + v2[i]) / 2
	}
	return v3
}

func (v Vector) Dot(v2 Vector) float64 {
	l := len(v)
	if len(v2) < l {
		l = len(v2)
	}
	var sum float64
	for i := 0; i < l; i++ {
		sum += v[i] * v2[i]
	}
	return sum
}

func (v Vector) SquareEucDistance(v2 Vector) float64 {
	l := len(v)
	if len(v2) < l {
		l = len(v2)
	}
	var sum float64
	for i := 0; i < l; i++ {
		sum += (v[i] - v2[i]) * (v[i] - v2[i])
	}
	return sum
}

func (v Vector) Hash() uint64 {
	var hasher hash.Hash64 = fnv.New64()
	for _, f := range v {
		binary.Write(hasher, binary.LittleEndian, f)
	}
	return hasher.Sum64()
}
