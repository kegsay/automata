package automata

type Trainer struct {
	Network      *Network
	Rate         float64
	Iterations   int
	MaxErrorRate float64
	CostFunction Coster
}

type TrainSet struct {
	Input  []float64
	Output []float64
}

func (t *Trainer) Train(trainingSet []TrainSet) error {
	// TODO: Cross-validation support
	for i := 0; i < t.Iterations; i++ {
		errorSum, err := t.trainSet(trainingSet, t.Rate, t.CostFunction)
		if err != nil {
			return err
		}
		errRate := errorSum / float64(len(trainingSet))
		if errRate < t.MaxErrorRate {
			return nil
		}
	}
	return nil
}

func (t *Trainer) trainSet(set []TrainSet, rate float64, coster Coster) (float64, error) {
	var errorSum float64
	for _, s := range set {
		actualOutput, err := t.Network.Activate(s.Input)
		if err != nil {
			return 0, err
		}
		t.Network.Propagate(rate, s.Output)
		errorSum += coster.Cost(s.Output, actualOutput)
	}
	return errorSum, nil
}
