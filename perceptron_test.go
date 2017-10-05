package automata_test

import (
	"github.com/kegsay/automata"
	"testing"
)

func TestPerceptron(t *testing.T) {
	// Train an XOR
	perceptron, err := automata.NewPerceptron([]int{2, 3, 1})
	if err != nil {
		t.Errorf("Failed to create NewPerceptron: %s", err.Error())
	}

	// Train it.
	trainer := automata.Trainer{
		Network:      perceptron,
		MaxErrorRate: 0.001,
		LearnRate:    0.1,
		Iterations:   10000,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	err = trainer.Train([]automata.TrainSet{
		{[]float64{0, 0}, []float64{0}},
		{[]float64{0, 1}, []float64{1}},
		{[]float64{1, 0}, []float64{1}},
		{[]float64{1, 1}, []float64{0}},
	})
	if err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

	// Test it.
	activateNetwork(t, perceptron, []float64{0, 0}, []float64{0})
	activateNetwork(t, perceptron, []float64{0, 1}, []float64{1})
	activateNetwork(t, perceptron, []float64{1, 0}, []float64{1})
	activateNetwork(t, perceptron, []float64{1, 1}, []float64{0})
}
