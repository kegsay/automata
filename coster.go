package automata

import (
	"math"
)

// Coster defines a cost function for use with the Trainer.
type Coster interface {
	// Cost of the given output. The length of target and output will always be the same.
	Cost(target, output []float64) (cost float64)
}

// MeanSquaredErrorCost implements the MSE cost function.
type MeanSquaredErrorCost struct{}

// Cost of the given output.
func (c *MeanSquaredErrorCost) Cost(target, output []float64) (cost float64) {
	for i := range output {
		cost += math.Pow(target[i]-output[i], 2)
	}
	return cost / float64(len(output))
}

// CrossEntropyCost implement the cross entropy function (Eq. 9)
type CrossEntropyCost struct{}

// Cost of the given output
func (c *CrossEntropyCost) Cost(target, output []float64) (cost float64) {
	nudge := 1e-15 // nudge all values up a little from 0-1 to make it impossible to do math.Log(0) which = -Inf
	for i := range output {
		n := (target[i] * math.Log(output[i]+nudge)) +
			((1 - target[i]) * math.Log((1+nudge)-output[i]))
		cost -= n
	}
	return
}
