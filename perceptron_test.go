package automata_test

import (
	"github.com/Kegsay/automata"
	"math"
	"math/rand"
	"testing"
)

func TestPerceptronXOR(t *testing.T) {
	testLookupTable := &automata.LookupTable{}
	rand.Seed(1) // consistent seed for consistent errors!
	perceptron, err := automata.NewPerceptronNetwork(testLookupTable, []int{2, 3, 1})
	if err != nil {
		t.Fatalf("Failed to create NewPerceptronNetwork: %s", err.Error())
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

func TestPerceptronSine(t *testing.T) {
	testLookupTable := &automata.LookupTable{}
	perceptron, err := automata.NewPerceptronNetwork(testLookupTable, []int{1, 12, 1})
	if err != nil {
		t.Fatalf("Failed to create NewPerceptronNetwork: %s", err.Error())
	}

	// keep sin output positive
	sinFunc := func(in float64) float64 {
		return (math.Sin(in) + 1) / 2
	}

	// Train it.
	trainer := automata.Trainer{
		Network:      perceptron,
		MaxErrorRate: 1e-6,
		LearnRate:    0.2,
		Iterations:   700,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	var ts []automata.TrainSet
	for i := 0; i < 500; i++ {
		rads := rand.Float64() * math.Pi * 2 // random radians
		ts = append(ts, automata.TrainSet{
			Input:  []float64{rads},
			Output: []float64{sinFunc(rads)},
		})
	}
	err = trainer.Train(ts)
	if err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

	// Test it.
	inputs := []float64{0, 0.5 * math.Pi, 2}
	outputs := []float64{sinFunc(0), sinFunc(0.5 * math.Pi), sinFunc(2)}
	for i := 0; i < len(inputs); i++ {
		output, err := perceptron.Activate([]float64{inputs[i]})
		if err != nil {
			t.Errorf("Activate(%v) returned an error: %s", inputs[i], err.Error())
		}
		delta := math.Abs(output[0] - outputs[i])
		if delta > 0.05 {
			t.Errorf("Activate(%v) returned %v, want %v with delta < 0.05", inputs[i], output[0], outputs[i])
		}
	}
}
