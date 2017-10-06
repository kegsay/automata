package automata

type Hopfield struct {
	Network
}

// Activate activates the network with the given input. The output values will always be either 0 or 1 to reflect
// the semantics of hopfield networks.
func (h *Hopfield) Activate(input []float64) ([]float64, error) {
	output, err := h.Network.Activate(input)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(output); i++ {
		if output[i] < 0.5 {
			output[i] = 0
		} else {
			output[i] = 1
		}
	}
	return output, nil
}

// NewHopfieldNetwork creates a new Hopfield network. 'size' dictates the number of input neurons and the number
// of output neurons. Generally this value should match the number of boolean options in your network
// e.g. 1 neuron per pixel if recalling images.
func NewHopfieldNetwork(size int) *Hopfield {
	inputLayer := NewLayer(size)
	outputLayer := NewLayer(size)
	inputLayer.Project(&outputLayer, LayerTypeAllToAll)
	return &Hopfield{
		Network: Network{
			Input:  &inputLayer,
			Output: &outputLayer,
		},
	}
}
