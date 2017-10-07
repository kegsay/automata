package automata

// NewLSTM creates a new Long-Short Term Memory network. 'inputSize' is the number of neurons in the input layer.
// 'outputSize' is the number of neurons in the output layer. The length of 'memoryBlocks' determines how many
// memory blocks the network will have, whilst the elements determine the number of memory cells in that block.
func NewLSTM(inputSize int, memoryBlocks []int, outputSize int) *Network {
	inputLayer := NewLayer(inputSize)
	outputLayer := NewLayer(outputSize)
	var hiddenLayers []Layer

	var prevMemoryBlock *Layer
	for _, cellSize := range memoryBlocks {
		inputGate := NewLayer(cellSize)
		inputGate.SetBias(1)
		forgetGate := NewLayer(cellSize)
		forgetGate.SetBias(1)
		outputGate := NewLayer(cellSize)
		outputGate.SetBias(1)
		memoryCell := NewLayer(cellSize)

		hiddenLayers = append(hiddenLayers, inputGate)
		hiddenLayers = append(hiddenLayers, forgetGate)
		hiddenLayers = append(hiddenLayers, memoryCell)
		hiddenLayers = append(hiddenLayers, outputGate)

		// connections from input
		input := inputLayer.Project(&memoryCell, LayerTypeAuto)
		inputLayer.Project(&inputGate, LayerTypeAuto)
		inputLayer.Project(&forgetGate, LayerTypeAuto)
		inputLayer.Project(&outputGate, LayerTypeAuto)

		// connections from prev memory block
		var cell *LayerConnection
		if prevMemoryBlock != nil {
			cell = prevMemoryBlock.Project(&memoryCell, LayerTypeAuto)
			prevMemoryBlock.Project(&inputGate, LayerTypeAuto)
			prevMemoryBlock.Project(&forgetGate, LayerTypeAuto)
			prevMemoryBlock.Project(&outputGate, LayerTypeAuto)
		}

		// connections from memory cell
		output := memoryCell.Project(&outputLayer, LayerTypeAuto)
		self := memoryCell.Project(&memoryCell, LayerTypeAuto)

		// peepholes
		memoryCell.Project(&inputGate, LayerTypeAllToAll)
		memoryCell.Project(&forgetGate, LayerTypeAllToAll)
		memoryCell.Project(&outputGate, LayerTypeAllToAll)

		// gates
		inputGate.Gate(input, GateTypeInput)
		forgetGate.Gate(self, GateTypeOneToOne)
		outputGate.Gate(output, GateTypeOutput)
		if cell != nil {
			inputGate.Gate(cell, GateTypeInput)
		}

		prevMemoryBlock = &memoryCell
	}

	// connect input layer to output layer (TODO: customise these conns?)
	inputLayer.Project(&outputLayer, LayerTypeAuto)

	return &Network{
		Input:  &inputLayer,
		Hidden: hiddenLayers,
		Output: &outputLayer,
	}
}
