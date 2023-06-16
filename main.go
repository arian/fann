package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/arian/fann/fann"
	"github.com/arian/fann/vector"
)

type store struct {
	invertedIndex map[string]int
	words         []string
	vectors       []vector.Vector
}

func loadVectors(fname string) (*store, error) {

	f, err := os.Open(fname)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	scanner := bufio.NewScanner(f)
	scanner.Split(bufio.ScanLines)

	// skip first line
	scanner.Scan()

	var vectors []vector.Vector
	var words []string
	inverted := make(map[string]int)

	i := 0
	for scanner.Scan() {
		if i >= 200_000 {
			break
		}

		line := scanner.Text()
		tokens := strings.Split(strings.Trim(line, " \n\r"), " ")
		word := tokens[0]
		tokens = tokens[1:]
		items := make([]float64, len(tokens))
		for i, token := range tokens {
			v, err := strconv.ParseFloat(token, 64)
			if err != nil {
				return nil, err
			}
			items[i] = v
		}

		vectors = append(vectors, items)
		words = append(words, word)
		inverted[word] = i

		i++

	}

	return &store{
		invertedIndex: inverted,
		words:         words,
		vectors:       vectors,
	}, nil
}

func main() {

	inputPath := flag.String("input-vec-path", "", "path to the wiki-news-300d-1M.vec file")
	flag.Parse()

	if *inputPath == "" {
		fmt.Println("input-vec-path is required")
		os.Exit(1)
	}

	t := time.Now()
	store, err := loadVectors(*inputPath)
	if err != nil {
		panic(err)
	}
	fmt.Printf("parsed vectors in %v\nFirst vector: %v\n", time.Since(t), store.vectors[0])
	t = time.Now()

	index := fann.NewANNIndex(
		3,
		8,
		store.vectors,
	)

	fmt.Printf("indexed %d vectors in %v \n", len(store.vectors), time.Since(t))

	fmt.Println("Type a word to search for similar words (max 10):")

	scanner := bufio.NewScanner(os.Stdin)
	for scanner.Scan() {
		word := scanner.Text()
		id, ok := store.invertedIndex[word]
		if !ok {
			fmt.Printf("word %s not found\n", word)
			continue
		}
		fmt.Printf("word %s id %d\n", word, id)

		vector := store.vectors[id]
		found := index.SearchApproximate(vector, 10)
		fmt.Println("Similar words:")
		for _, id := range found {
			fmt.Printf("- %s\n", store.words[id])
		}
	}

	if err := scanner.Err(); err != nil {
		fmt.Fprintln(os.Stderr, "reading standard input:", err)
	}
}
