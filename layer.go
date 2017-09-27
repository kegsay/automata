package automata

import "fmt"

// Layer represents a group of neurons which activate together.
type Layer struct {
	List        []Neuron
	ConnectedTo []LayerConnection
}

func NewLayer(size int) Layer {
	neurons := make([]Neuron, size)
	for i := 0; i < size; i++ {
		neurons[i] = NewNeuron()
	}
	return Layer{
		List: neurons,
	}
}

// Activate all neurons in the layer.
func (l *Layer) Activate(inputs []float64) ([]float64, error) {
	var activations []float64

	// Activate without an input
	if inputs == nil {
		for i := 0; i < len(l.List); i++ {
			activation := l.List[i].Activate(nil)
			activations = append(activations, activation)
		}
	} else if len(inputs) != len(l.List) {
		return nil, fmt.Errorf("input and layer size mismatch: cannot activate")
	} else { // Activate with input
		for i := 0; i < len(l.List); i++ {
			activation := l.List[i].Activate(&inputs[i])
			activations = append(activations, activation)
		}
	}

	return activations, nil
}

// Propagate an error on all neurons in this layer.
func (l *Layer) Propagate(rate float64, target []float64) error {
	if target != nil {
		if len(target) != len(l.List) {
			return fmt.Errorf("target and layer size mismatch: cannot propagate")
		}
		for i := len(l.List) - 1; i >= 0; i-- {
			l.List[i].Propagate(rate, &target[i])
		}
	} else {
		for i := len(l.List) - 1; i >= 0; i-- {
			l.List[i].Propagate(rate, nil)
		}
	}
	return nil
}

// Project a connection from this layer to another one.
func (l *Layer) Project(toLayer *Layer, ltype LayerType) *LayerConnection {
	if l.isConnected(toLayer) {
		return nil
	}
	lc := NewLayerConnection(l, toLayer, ltype)
	return &lc
}

// Gate a connection between two layers.
func (l *Layer) Gate() {}

// isConnected returns true if this layer is connected to the target layer already.
func (l *Layer) isConnected(targetLayer *Layer) bool {
	for _, neuron := range l.List {
		for _, target := range targetLayer.List {
			if neuron.ConnectionForNeuron(&target) != nil {
				return true
			}
		}
	}
	return false
}
