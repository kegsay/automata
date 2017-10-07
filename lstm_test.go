package automata_test

import (
	"github.com/kegsay/automata"
	"math/rand"
	"strings"
	"testing"
	"time"
)

// Test the LSTM by giving it an Embedded Reber Grammar test.
//
// We'll use the example at https://www.willamette.edu/~gorr/classes/cs449/reber.html : the ERG image is
// contained in this repository.
func TestLSTMERG(t *testing.T) {
	rand.Seed(time.Now().UnixNano())

	// enough memory cells for each node in the ERG
	lstm := automata.NewLSTM(7, []int{6, 6, 6}, 7)

	ergTrainingStrings := make([]string, 12)
	for i := 0; i < 12; i++ {
		ergTrainingStrings[i] = generateERG(60)
	}

	trainer := automata.Trainer{
		Network:      lstm,
		MaxErrorRate: 0.01,
		LearnRate:    0.5,
		Iterations:   5,
		CostFunction: &automata.MeanSquaredErrorCost{},
	}
	// The point is for the ANN to predict the next letter in the ERG, an to know when the sequence
	// has been restarted (reached end and started a new one)
	ergTrain := strings.Join(ergTrainingStrings, "")
	trainingSet := make([]automata.TrainSet, len(ergTrain)-1)
	for i := 0; i < len(ergTrain)-1; i++ {
		trainingSet[i] = automata.TrainSet{
			Input:  letterToInput(t, rune(ergTrain[i])),
			Output: letterToInput(t, rune(ergTrain[i+1])),
		}
	}
	err := trainer.Train(trainingSet)
	if err != nil {
		t.Fatalf("trainer.Train threw error: %s", err.Error())
	}

}

// Convert the letter into a suitable form for consumption in the network. Basically it's a massive set of flags,
// one for each letter. Flag is on if letter is selected.
func letterToInput(t *testing.T, letter rune) []float64 {
	switch letter {
	case 'B':
		return []float64{1, 0, 0, 0, 0, 0, 0}
	case 'T':
		return []float64{0, 1, 0, 0, 0, 0, 0}
	case 'X':
		return []float64{0, 0, 1, 0, 0, 0, 0}
	case 'P':
		return []float64{0, 0, 0, 1, 0, 0, 0}
	case 'V':
		return []float64{0, 0, 0, 0, 1, 0, 0}
	case 'S':
		return []float64{0, 0, 0, 0, 0, 1, 0}
	case 'E':
		return []float64{0, 0, 0, 0, 0, 0, 1}
	default:
		t.Fatalf("Invalid letter: %s", letter)
	}
	return nil
}

// convert the output back into a letter
func outputToLetter(output []float64) string {
	flagIndex := -1
	for i := range output {
		val := round(output[i])
		if val == 0 {
			continue
		}
		if flagIndex != -1 {
			// multiple flags were set, which makes no sense. Return a marker to indicate this.
			return "1"
		}
		flagIndex = i
	}

	switch flagIndex {
	case 0:
		return "B"
	case 1:
		return "T"
	case 2:
		return "X"
	case 3:
		return "P"
	case 4:
		return "V"
	case 5:
		return "S"
	case 6:
		return "E"
	}
	return "0" // no flags set
}

// Numbers are states in the finite state machine.
// Letters are the grammar produced during state transitions.
//
//             S
//             ||
//     +--T-->[1]---X--->[2]--S-+
//     |                 | ^    |
//-B->[3]      +----X----+ P    +-->[4]--E-->
//     |       V           |    |
//     +--P-->[5]---V--->[6]--V-+
//             ||
//             T
func generateEmbeddedPart(length int) string { // BT   B  TXXTTVPXVPXVPX  E  TE
	if length < 5 {
		panic("Cannot produce ERG less than 5 chars long!")
	}
	state := 3
	length -= 2 // remove B and E
	var output []string

	// state# -> count of transitions to finish the grammar
	minStepsToStateFour := make(map[int]int)
	minStepsToStateFour[3] = 3
	minStepsToStateFour[1] = 2
	minStepsToStateFour[5] = 2
	minStepsToStateFour[2] = 1
	minStepsToStateFour[6] = 1

loop:
	for i := 0; i < length; i++ {
		stepsLeft := length - i
		if minStepsToStateFour[state] == stepsLeft {
			// we have to leave now, go directly to state [4]
			switch state {
			case 2:
				output = append(output, "S")
			case 6:
				output = append(output, "V")
			case 1:
				output = append(output, "XS")
			case 5:
				output = append(output, "VV")
			case 3:
				if rand.Float64() < 0.5 {
					output = append(output, "TXS")
				} else {
					output = append(output, "PVV")
				}
			}
			break loop
		} else {
			val := rand.Float64()
			switch state {
			case 1:
				if val < 0.5 {
					output = append(output, "S") // state remains unchanged
				} else {
					output = append(output, "X")
					state = 2
				}
			case 2:
				if minStepsToStateFour[5] > (stepsLeft - 1) {
					// if we move to [5] then we won't be able to leave in time, so leave now.
					output = append(output, "S")
					break loop
				}
				// we have to transition to 5 as we are forced to leave if we transition to 4
				output = append(output, "X")
				state = 5
			case 3:
				if val < 0.5 {
					output = append(output, "T")
					state = 1
				} else {
					output = append(output, "P")
					state = 5
				}
			case 5:
				if val < 0.5 {
					output = append(output, "T") // state remains unchanged
				} else {
					output = append(output, "V")
					state = 6
				}
			case 6:
				if minStepsToStateFour[2] > (stepsLeft - 1) {
					// if we move to [2] then we won't be able to leave in time, so leave now.
					output = append(output, "V")
					break loop
				}
				// we have to transition to 2 as we are forced to leave if we transition to 4
				output = append(output, "P")
				state = 2
			}
		}
	}

	return "B" + strings.Join(output, "") + "E"
}

func generateERG(length int) string { // length is not guaranteed due to state tranisitions, will be mostly correct.
	output := []string{
		"B",
		pickRune([]rune{'T', 'P'}),
		generateEmbeddedPart(length - 4),
	}
	if output[1] == "T" {
		output = append(output, "T")
	} else {
		output = append(output, "P")
	}
	output = append(output, "E")
	return strings.Join(output, "")
}

// pick a random rune from the list, equal chance.
func pickRune(opts []rune) string {
	odds := float64(1 / float64(len(opts))) // this is the increment amount e.g 0.25 if 4 options
	r := rand.Float64()                     // random value between 0 ~ 1
	for i := range opts {
		// i=0, odds=0.25 - r is between 0 ~ 0.25 (0 inclusive)
		// i=1, odds=0.25 - r is between 0.25 ~ 0.5 (0.25 inclusive)
		// i=2, odds=0.25 - r is between 0.5 ~ 0.75 (0.5 inclusive)
		// i=3, odds=0.25 - r is between 0.75 ~ 1 (0.75 inclusive)
		if r >= (float64(i)*odds) && r < (float64(i+1)*odds) {
			return string(opts[i])
		}
	}
	return string(opts[len(opts)-1]) // r=1
}
