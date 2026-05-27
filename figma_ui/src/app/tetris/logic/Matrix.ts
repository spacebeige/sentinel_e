export function createMatrix(
width = 10,
height = 20
) {
return Array.from({
length: height
}).map(() =>
Array(width).fill(0)
)
}
