package automata

import (
	"math"
	"testing"
)

func TestANDGate(t *testing.T) {
	// Make the network.
	inputLayer := NewLayer(2)
	outputLayer := NewLayer(1)
	inputLayer.Project(&outputLayer, LayerTypeAuto)
	network := Network{
		Input:  &inputLayer,
		Hidden: nil,
		Output: &outputLayer,
	}

	// Train it.
	trainer := Trainer{
		Network:      &network,
		MaxErrorRate: 0.001,
		Iterations:   1000,
		CostFunction: &MeanSquaredErrorCost{},
	}
	err := trainer.Train([]TrainSet{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{0}},
		{[]float64{1, 0}, []float64{0}},
		{[]float64{1, 1}, []float64{1}},
	})
	if err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

	// Test it.
	activateNetwork(t, &network, []float64{0, 0}, []float64{0})
	activateNetwork(t, &network, []float64{0, 1}, []float64{0})
	activateNetwork(t, &network, []float64{1, 0}, []float64{0})
	activateNetwork(t, &network, []float64{1, 1}, []float64{1})
}

func activateNetwork(t *testing.T, network *Network, input, desiredOutput []float64) {
	output, err := network.Activate(input)
	if err != nil {
		t.Errorf("%v returned an error: %s", input, err.Error())
	}
	for i, out := range output {
		if round(out) != desiredOutput[i] {
			t.Errorf("%v : want %f, got %f (%f)", input, desiredOutput[i], round(out), out)
		}
	}
}

func round(in float64) float64 {
	return math.Floor(in + 0.5)
}
