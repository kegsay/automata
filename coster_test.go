package automata

import (
	"testing"
)

func TestMeanSquaredErrorCost(t *testing.T) {
	mse := MeanSquaredErrorCost{}
	output := mse.Cost([]float64{0, 0.5, 1}, []float64{1, 0.5, 0})
	want := 0.66666666666666666666
	if output != want {
		t.Errorf("Cost([0, 0.5, 1], [1, 0.5, 0]) : want %f, got %f", want, output)
	}

	output = mse.Cost([]float64{0, 0.5, 1}, []float64{0, 0.5, 1})
	want = 0
	if output != want {
		t.Errorf("Cost([0, 0.5, 1], [0, 0.5, 1]) : want %f, got %f", want, output)
	}
}

func TestCrossEntropyCost(t *testing.T) {
	ce := CrossEntropyCost{}
	output := ce.Cost([]float64{0, 0.5, 1}, []float64{1, 0.5, 0})
	want := 69.66613905215368
	if output != want {
		t.Errorf("Cost([0, 0.5, 1], [1, 0.5, 0]) : want %f, got %f", want, output)
	}

	output = ce.Cost([]float64{0, 0.5, 1}, []float64{0, 0.5, 1})
	want = 0.693147180559941
	if output != want {
		t.Errorf("Cost([0, 0.5, 1], [0, 0.5, 1]) : want %f, got %f", want, output)
	}
}

func TestBinaryCost(t *testing.T) {
	bc := BinaryCost{}
	output := bc.Cost([]float64{0, 0.5, 1}, []float64{1, 0.5, 0})
	want := 2.0
	if output != want {
		t.Errorf("Cost([0, 0.5, 1], [1, 0.5, 0]) : want %f, got %f", want, output)
	}

	output = bc.Cost([]float64{0, 0.5, 1}, []float64{0, 0.5, 1})
	want = 0.0
	if output != want {
		t.Errorf("Cost([0, 0.5, 1], [0, 0.5, 1]) : want %f, got %f", want, output)
	}
}
