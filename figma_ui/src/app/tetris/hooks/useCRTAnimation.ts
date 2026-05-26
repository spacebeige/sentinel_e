
import { useEffect } from "react"

export function useCRTAnimation() {
useEffect(() => {
const interval = setInterval(() => {}, 1000)


return () => clearInterval(interval)


}, [])
}
