import { useState } from "react"
import { createInitialState } from "../logic/Engine"

export function useTetrisEngine() {
const [state] = useState(
createInitialState()
)

return state
}
