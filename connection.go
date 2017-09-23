package automata

import "fmt"

var connCount = 0

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

func NewConnection(from, to Neuron, weight float64) Connection {
	return Connection{
		ID:     connUID(),
		From:   &from,
		To:     &to,
		Weight: weight,
	}
}

type ConnMap map[ConnID]Connection

type LayerType int

const (
	LayerTypeAuto LayerType = iota
	LayerTypeAllToAll
	LayerTypeOneToOne
	LayerTypeAllToElse
)

// LayerConnection represents a connection between two layers.
type LayerConnection struct {
	ID          ConnID
	From        Layer
	To          Layer
	Type        LayerType
	Weights     []float64
	Connections ConnMap
	List        []Connection
}

func NewLayerConnection(from, to Layer, ltype LayerType, weights []float64) LayerConnection {
	if ltype == LayerTypeAuto {
		if &from == &to {
			ltype = LayerTypeOneToOne
		} else {
			ltype = LayerTypeAllToAll
		}
	}

	var list []Connection
	connsByID := make(ConnMap)
	switch ltype {
	case LayerTypeOneToOne:
		// A neuron in position i in the 'from' layer gets projected to the matching neuron in position i
		// in the 'to' layer and nothing more. No neuron is supplied if there is no matching neuron.
		for i, neuron := range from.List {
			var toNeuron *Neuron
			if i < len(to.List) {
				toNeuron = &to.List[i]
			}
			conn := neuron.Project(toNeuron, weights)
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
				conn := fromNeuron.Project(&toNeuron, weights)
				connsByID[conn.ID] = conn
				list = append(list, conn)
			}
		}
	default:
		panic(fmt.Sprintf("NewLayerConnection: unknown layer type %d", ltype))
	}

	lc := LayerConnection{
		ID:          connUID(),
		From:        from,
		To:          to,
		Type:        ltype,
		Weights:     weights,
		Connections: connsByID,
		List:        list,
	}
	from.ConnectedTo = append(from.ConnectedTo, lc)

	return lc
}

func connUID() ConnID {
	connCount += 1
	return ConnID(connCount)
}
