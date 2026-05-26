import React from 'react';
import GameScene from "../game/GameScene";

export function ExplorePage() {
  return (
    <div className="w-screen h-screen overflow-hidden bg-black" style={{ position: "fixed", top: 0, left: 0, zIndex: 50 }}>
      <GameScene />
    </div>
  );
}
