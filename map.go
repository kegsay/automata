package automata

// NeuronMap provides an efficient way of accessing Neurons with NeuronIDs.
type NeuronMap []*Neuron

func (nm NeuronMap) Get(index NeuronID) *Neuron {
	if int(index) > (len(nm) - 1) {
		return nil
	}
	return nm[index]
}

func (nm NeuronMap) Set(parent *Neuron, index NeuronID, neuron *Neuron) {
	if int(index) > (len(nm) - 1) {
		diff := int(index) - (len(nm) - 1)
		nm = append(nm, make([]*Neuron, diff)...)
		parent.Neighbours = nm
	}
	nm[index] = neuron
}
