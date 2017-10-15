package automata

import "fmt"
import "math/rand"

type LayerType int

const (
	LayerTypeAuto LayerType = iota
	LayerTypeAllToAll
	LayerTypeOneToOne
	LayerTypeAllToElse
)

type ConnID int64

// Connection represents a connection between two neurons.
type Connection struct {
	ID     ConnID
	From   *Neuron
	To     *Neuron
	Gater  *Neuron
	Weight float64
	Gain   float64
}

func NewConnection(from, to *Neuron, weight *float64) *Connection {
	if weight == nil {
		w := (rand.Float64() * 0.2) - 0.1 // random weight between -0.1 and +0.1
		weight = &w
	}
	conn := &Connection{
		From:   from,
		To:     to,
		Weight: *weight,
		Gain:   1,
	}
	id := GlobalLookupTable.SetConnection(conn)
	conn.ID = id
	return conn
}

// LayerConnection represents a connection between two layers.
type LayerConnection struct {
	From        *Layer
	To          *Layer
	Type        LayerType
	Connections map[ConnID]*Connection
	List        []*Connection
}

func NewLayerConnection(from, to *Layer, ltype LayerType) LayerConnection {
	if ltype == LayerTypeAuto {
		if from == to {
			ltype = LayerTypeOneToOne
		} else {
			ltype = LayerTypeAllToAll
		}
	}

	var list []*Connection
	connsByID := make(map[ConnID]*Connection)
	switch ltype {
	case LayerTypeOneToOne:
		// A neuron in position i in the 'from' layer gets projected to the matching neuron in position i
		// in the 'to' layer and nothing more. No neuron is supplied if there is no matching neuron.
		for i, neuron := range from.List {
			var toNeuron *Neuron
			if i < len(to.List) {
				toNeuron = to.List[i]
			}
			conn := neuron.Project(toNeuron, nil)
			connsByID[conn.ID] = conn
			list = append(list, conn)
		}
	case LayerTypeAllToAll:
		fallthrough
	case LayerTypeAllToElse:
		// Each neuron in the 'from' layer gets projected to all neurons in the 'to' layer.
		// 'ToElse' stops the neuron projecting to itself if it exists in the 'to' layer.
		// 'ToAll' projects to all neurons regardless.
		for _, fromNeuron := range from.List {
			for _, toNeuron := range to.List {
				if ltype == LayerTypeAllToElse && &fromNeuron == &toNeuron {
					continue
				}
				conn := fromNeuron.Project(toNeuron, nil)
				connsByID[conn.ID] = conn
				list = append(list, conn)
			}
		}
	default:
		panic(fmt.Sprintf("NewLayerConnection: unknown layer type %d", ltype))
	}

	lc := LayerConnection{
		//ID:          connUID(),
		From:        from,
		To:          to,
		Type:        ltype,
		Connections: connsByID,
		List:        list,
	}
	from.ConnectedTo = append(from.ConnectedTo, lc)

	return lc
}
