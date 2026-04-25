"use client";

import { useEffect } from "react";
import { SuperTokensWrapper } from "supertokens-auth-react";
import { initSuperTokens } from "../lib/supertokens";

export default function Providers({ children }) {
  useEffect(() => {
    initSuperTokens();
  }, []);

  return <SuperTokensWrapper>{children}</SuperTokensWrapper>;
}
