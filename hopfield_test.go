package automata_test

import (
	"github.com/Kegsay/automata"
	"strings"
	"testing"
)

const hopfieldImageOne = `
0001000
0001000
0001000
0001000
0001000
0001000
0001000`

const hopfieldImageTwo = `
1111111
0000001
0000001
1111111
1000000
1000000
1111111`

const hopfieldImageThree = `
1111111
0000001
0000001
1111111
0000001
0000001
1111111`

const hopfieldImageFour = `
1000001
1000001
1000001
1111111
0000001
0000001
0000001`

const hopfieldImageFive = `
1111111
1000000
1000000
1111111
0000001
0000001
1111111`

func TestHopfieldImages(t *testing.T) {
	testLookupTable := &automata.LookupTable{}
	gridSize := 7
	hopfield := automata.NewHopfieldNetwork(testLookupTable, gridSize*gridSize) // 7x7 grid

	// Train it with "images" of numbers 1 -> 5.
	trainer := automata.Trainer{
		Network:      hopfield,
		MaxErrorRate: 0.00001,
		LearnRate:    1,
		Iterations:   10000,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	trainingSet := []automata.TrainSet{
		imageToTrainSet(t, hopfieldImageOne),
		imageToTrainSet(t, hopfieldImageTwo),
		imageToTrainSet(t, hopfieldImageThree),
		imageToTrainSet(t, hopfieldImageFour),
		imageToTrainSet(t, hopfieldImageFive),
	}

	err := trainer.Train(trainingSet)
	if err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

	// test it with a serif one
	activateNetwork(
		t, &hopfield.Network, imageToInput(t, `
0011000
0001000
0001000
0001000
0001000
0001000
0011100`), imageToInput(t, hopfieldImageOne))

	// test it with a two
	activateNetwork(
		t, &hopfield.Network, imageToInput(t, `
0011100
1100011
0000010
0111000
1000000
1000001
0111110`), imageToInput(t, hopfieldImageTwo))

	// test it with a three
	activateNetwork(
		t, &hopfield.Network, imageToInput(t, `
0111110
1000001
0000001
0011110
0000001
0000001
0011110`), imageToInput(t, hopfieldImageThree))

	// test it with a four
	activateNetwork(
		t, &hopfield.Network, imageToInput(t, `
1000001
0100001
0010010
0011110
0000001
0000001
0000010`), imageToInput(t, hopfieldImageFour))

	// test it with a five
	activateNetwork(
		t, &hopfield.Network, imageToInput(t, `
1111110
1100000
1100000
0111110
0000001
1000001
0111110`), imageToInput(t, hopfieldImageFive))

}

func imageToTrainSet(t *testing.T, img string) automata.TrainSet {
	input := imageToInput(t, img)
	return automata.TrainSet{
		Input:  input,
		Output: input,
	}
}

func imageToInput(t *testing.T, img string) []float64 {
	// strip new lines
	img = strings.Replace(img, "\r", "", -1)
	img = strings.Replace(img, "\n", "", -1)

	// convert 0 and 1 to actual floats and return
	set := make([]float64, len(img))
	for i := 0; i < len(img); i++ {
		if img[i] == '0' {
			set[i] = float64(0)
		} else if img[i] == '1' {
			set[i] = float64(1)
		} else {
			t.Fatalf("imageToTrainSet: bad test image char: %q", img[i])
		}
	}
	return set
}
