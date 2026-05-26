export function calculateCoherence(
lines: number
) {
return lines * 100
}

export function calculateDivergence(
holes: number
) {
return Math.min(100, holes * 3)
}
