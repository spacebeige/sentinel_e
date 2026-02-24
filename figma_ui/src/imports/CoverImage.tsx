import imgCoverUiKitIOs26 from "figma:asset/d7c995ab95a09e294934c446f3ff248fc18762c4.png";

export default function CoverImage() {
  return (
    <div className="relative size-full" data-name="Cover image">
      <div className="absolute h-[1080px] left-0 top-0 w-[1920px]" data-name="Cover - UI Kit - iOS 26">
        <img alt="" className="absolute inset-0 max-w-none object-cover pointer-events-none size-full" src={imgCoverUiKitIOs26} />
      </div>
    </div>
  );
}