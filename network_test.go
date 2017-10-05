package automata_test

import (
	"github.com/kegsay/automata"
	"math"
	"testing"
)

func TestANDGate(t *testing.T) {
	// Make the network.
	inputLayer := automata.NewLayer(2)
	outputLayer := automata.NewLayer(1)
	inputLayer.Project(&outputLayer, automata.LayerTypeAuto)
	network := automata.Network{
		Input:  &inputLayer,
		Hidden: nil,
		Output: &outputLayer,
	}

	// Train it.
	trainer := automata.Trainer{
		Network:      &network,
		MaxErrorRate: 0.001,
		LearnRate:    0.1,
		Iterations:   1000,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	err := trainer.Train([]automata.TrainSet{
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

func TestORGate(t *testing.T) {
	// Make the network.
	inputLayer := automata.NewLayer(2)
	outputLayer := automata.NewLayer(1)
	inputLayer.Project(&outputLayer, automata.LayerTypeAuto)
	network := automata.Network{
		Input:  &inputLayer,
		Hidden: nil,
		Output: &outputLayer,
	}

	// Train it.
	trainer := automata.Trainer{
		Network:      &network,
		MaxErrorRate: 0.001,
		LearnRate:    0.1,
		Iterations:   1000,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	err := trainer.Train([]automata.TrainSet{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{1}},
	})
	if err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

	// Test it.
	activateNetwork(t, &network, []float64{0, 0}, []float64{0})
	activateNetwork(t, &network, []float64{0, 1}, []float64{1})
	activateNetwork(t, &network, []float64{1, 0}, []float64{1})
	activateNetwork(t, &network, []float64{1, 1}, []float64{1})
}

func TestNOTGate(t *testing.T) {
	// Make the network.
	inputLayer := automata.NewLayer(1)
	outputLayer := automata.NewLayer(1)
	inputLayer.Project(&outputLayer, automata.LayerTypeAuto)
	network := automata.Network{
		Input:  &inputLayer,
		Hidden: nil,
		Output: &outputLayer,
	}

	// Train it.
	trainer := automata.Trainer{
		Network:      &network,
		MaxErrorRate: 0.001,
		LearnRate:    0.1,
		Iterations:   1000,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	err := trainer.Train([]automata.TrainSet{
		{[]float64{0}, []float64{1}},
		{[]float64{1}, []float64{0}},
	})
	if err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

	// Test it.
	activateNetwork(t, &network, []float64{0}, []float64{1})
	activateNetwork(t, &network, []float64{1}, []float64{0})
}

func activateNetwork(t *testing.T, network *automata.Network, input, desiredOutput []float64) {
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
