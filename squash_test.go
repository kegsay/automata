package automata

import (
	"math"
	"testing"
)

func TestSquashLogistic(t *testing.T) {
	s := SquashLogistic{}
	if out := s.Squash(0, false); out != 0.5 {
		t.Errorf("want 0.5, got %f", out)
	}
	if out := s.Squash(0, true); out != 0.25 {
		t.Errorf("want 0.25, got %f", out)
	}
}

func TestSquashTanh(t *testing.T) {
	s := SquashTanh{}
	if out := s.Squash(0.5, false); math.Abs(out-0.46211715726) > 0.000001 {
		t.Errorf("want 0.46211715726, got %f", out)
	}
	if out := s.Squash(0.5, true); math.Abs(out-0.786448) > 0.000001 {
		t.Errorf("want 0.786448, got %f", out)
	}
}

func TestSquashIdentity(t *testing.T) {
	s := SquashIdentity{}
	if out := s.Squash(50, false); out != 50 {
		t.Errorf("want 50, got %f", out)
	}
	if out := s.Squash(50, true); out != 1 {
		t.Errorf("want 1, got %f", out)
	}
}

func TestSquashRelu(t *testing.T) {
	s := SquashRelu{}
	if out := s.Squash(50, false); out != 50 {
		t.Errorf("want 50, got %f", out)
	}
	if out := s.Squash(50, true); out != 1 {
		t.Errorf("want 1, got %f", out)
	}

	if out := s.Squash(-50, false); out != 0 {
		t.Errorf("want 0, got %f", out)
	}
	if out := s.Squash(-50, true); out != 0 {
		t.Errorf("want 0, got %f", out)
	}
}
