package automata

type GateType int

const ( // TODO: Better construct than just an enum?
	GateTypeInput = iota
	GateTypeOutput
	GateTypeOneToOne
)
