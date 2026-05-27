//FILE:
///figma_ui/src/app/tetris/components/MemoryMeter.tsx

export default function MemoryMeter({
value,
}: {
value: number
}) {
return ( <div className="flex gap-2">
{Array.from({ length: 20 }).map((_, i) => (
<div
key={i}
className="w-[12px] h-[42px]"
style={{
background:
i < value
? "#0aff84"
: "#052d18",
boxShadow:
i < value
? "0 0 12px #0aff84"
: "none"
}}
/>
))} </div>
)
}
