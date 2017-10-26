package automata

// LookupTable stores mappings of:
//  - Neuron IDs to Neurons
//  - Connection IDs to Connections
// It can be thought of as a global hash map, but is implemented slightly differently for
// performance reasons.
//
// Rationale: Neurons need references to other neurons/connections (e.g. neighbours). The simplest
// way to do this is to store a map[ID]*Thing in the Neuron struct itself. This ends up being slow because
// this is called a lot and each time incurs hashing overheads. It would be better if this was done
// as a slice, especially since network topologies don't tend to change at runtime so there are no
// resizing overheads. This means the IDs would be indexes. LookupTable is a massive global
// slice which provides Neurons/Layers/Connections access to each other via their ID. It's not a true
// global variable as it is dependency injected at the point of use, allowing the ability of running
// multiple disconnected networks without sharing the same ID space. Sharing the same LookupTable
// for all neurons in a network also lowers memory usage from O(n) to O(1), as each neuron is not
// having to store its own mini lookup table.
type LookupTable struct {
	Neurons     []*Neuron
	Connections []*Connection
}

// SetNeuron in the lookup table. Returns the ID for this neuron.
func (t *LookupTable) SetNeuron(neuron *Neuron) NeuronID {
	t.Neurons = append(t.Neurons, neuron)
	return NeuronID(len(t.Neurons) - 1)
}

// GetNeuron from the lookup table. Returns nil if the ID does not exist in the table.
func (t *LookupTable) GetNeuron(id NeuronID) *Neuron {
	if int(id) > (len(t.Neurons) - 1) {
		return nil
	}
	return t.Neurons[id]
}

// SetConnection in the lookup table. Returns the ID for this connection.
func (t *LookupTable) SetConnection(conn *Connection) ConnID {
	t.Connections = append(t.Connections, conn)
	return ConnID(len(t.Connections) - 1)
}

// SetConnectionWithID in the lookup table. If the ID is already associated with a Connection then
// it is replaced.
func (t *LookupTable) SetConnectionWithID(id ConnID, conn *Connection) {
	if int(id) > (len(t.Connections) - 1) {
		// pad out the slice
		diff := int(id) - (len(t.Connections) - 1)
		t.Connections = append(t.Connections, make([]*Connection, diff)...)
	}
	t.Connections[id] = conn
}

// GetConnection from the lookup table. Returns nil if the ID does not exist in the table.
func (t *LookupTable) GetConnection(id ConnID) *Connection {
	if int(id) > (len(t.Connections) - 1) {
		return nil
	}
	return t.Connections[id]
}
