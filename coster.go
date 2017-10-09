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

// TODO: Cross-entropy (Eq. 9)
