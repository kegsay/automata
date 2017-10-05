package automata

import (
	"fmt"
)

// NewPerceptron creates a new Perceptron with the number of layers equal to the length of the provided slice.
// The elements in the slice determine the size of each layer. The first element is the input layer. The last
// element is the output layer. The inbetween elements are hidden layers. The length of the slice must be at
// least 3 to accomodate an input, hidden and output layer. For example:
//
//   // create a network with:
//   //  - 2 neurons on the input layer
//   //  - 3 neurons on the first hidden layer
//   //  - 4 neurons on the second hidden layer
//   //  - 1 neuron on the output layer
//   perceptron := automata.NewPerceptron([]int{2,3,4,1})
func NewPerceptron(sizesOfLayers []int) (*Network, error) {
	if len(sizesOfLayers) < 3 {
		return nil, fmt.Errorf("NewPerceptron: sizesOfLayers must be at least 3, got %d", len(sizesOfLayers))
	}
	inputLayer := NewLayer(sizesOfLayers[0])
	outputLayer := NewLayer(sizesOfLayers[len(sizesOfLayers)-1])
	prevLayer := &inputLayer
	var hiddenLayers []Layer
	for i := 1; i < len(sizesOfLayers)-1; i++ {
		hidden := NewLayer(sizesOfLayers[i])
		hiddenLayers = append(hiddenLayers, hidden)
		prevLayer.Project(&hidden, LayerTypeAuto)
		prevLayer = &hidden
	}
	prevLayer.Project(&outputLayer, LayerTypeAuto)
	return &Network{
		Input:  &inputLayer,
		Hidden: hiddenLayers,
		Output: &outputLayer,
	}, nil
}
