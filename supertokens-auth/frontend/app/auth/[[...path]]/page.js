"use client";

import { canHandleRoute, getRoutingComponent } from "supertokens-auth-react/ui";
import { ThirdPartyPreBuiltUI } from "supertokens-auth-react/recipe/thirdparty/prebuiltui";
import { initSuperTokens } from "../../../lib/supertokens";

initSuperTokens();

export default function AuthPage() {
  if (canHandleRoute([ThirdPartyPreBuiltUI])) {
    return getRoutingComponent([ThirdPartyPreBuiltUI]);
  }

  return null;
}
