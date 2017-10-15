package automata

import "math"

// Squasher implements a squashing function which can be used as an activation function.
// Squashing functions modify inputs allowing neurons to model non-linear relationships.
// This typically involves clamping/bounding the output at one or both ends.
//
// These functions can be very sensitive to the weights which are applied to them which
// can make training difficult because you need to discover the precise weightings which will
// squash the input in the way you want. Some functions can exacerbate the Vanishing Gradient
// problem.
//
// Common squash functions are already implemented.
type Squasher interface {
	// Squash an input x into a suitable output. If 'derivate' is true, the derivative should
	// be returned.
	Squash(x float64, derivate bool) float64
}

// SquashLogistic implements the logistic function.
type SquashLogistic struct{}

// Squashes x.
func (s *SquashLogistic) Squash(x float64, derivate bool) float64 {
	fx := 1.0 / (1.0 + math.Exp(-x))
	if !derivate {
		return fx
	}
	return fx * (1 - fx)
}

// SquashTanh implements tanh function.
type SquashTanh struct{}

// Squashes x.
func (s *SquashTanh) Squash(x float64, derivate bool) float64 {
	if derivate {
		return 1 - math.Pow(math.Tanh(x), 2)
	}
	return math.Tanh(x)
}

// SquashIdentity implements the identity function.
type SquashIdentity struct{}

// Squashes x.
func (s *SquashIdentity) Squash(x float64, derivate bool) float64 {
	if derivate {
		return 1
	}
	return x
}

// SquashReLU implements the ReLU activation function, which is not subject to the Vanishing
// Gradient problem
// .
// https://en.wikipedia.org/wiki/Rectifier_(neural_networks)
type SquashRelu struct{}

// Squashes x.
func (s *SquashRelu) Squash(x float64, derivate bool) float64 {
	if derivate {
		if x > 0 {
			return 1
		}
		return 0
	}
	if x > 0 {
		return x
	}
	return 0
}
