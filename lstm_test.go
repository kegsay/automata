package automata_test

import (
	"github.com/Kegsay/automata"
	"testing"
)

func TestLSTM_ShortTerm(t *testing.T) {
	lstm := automata.NewLSTM(1, []int{6}, 1)
	trainer := automata.Trainer{
		Network:      lstm,
		MaxErrorRate: 0.001,
		LearnRate:    0.2,
		Iterations:   10000,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	// LSTM must remember where in the sequence it is to produce the right output.
	trainSets := []automata.TrainSet{
		{[]float64{0}, []float64{0}},
		{[]float64{1}, []float64{1}},
		{[]float64{1}, []float64{0}},
		{[]float64{0}, []float64{1}},
		{[]float64{0}, []float64{0}},
	}

	if err := trainer.Train(trainSets); err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

	// test it
	activateNetwork(t, lstm, []float64{0}, []float64{0})
	activateNetwork(t, lstm, []float64{1}, []float64{1})
	activateNetwork(t, lstm, []float64{1}, []float64{0})
	activateNetwork(t, lstm, []float64{0}, []float64{1})
	activateNetwork(t, lstm, []float64{0}, []float64{0})

}
