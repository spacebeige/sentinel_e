"use client";

import SuperTokens from "supertokens-auth-react";
import ThirdParty, { Google, Github } from "supertokens-auth-react/recipe/thirdparty";
import Session from "supertokens-auth-react/recipe/session";

let initialized = false;

export function initSuperTokens() {
  if (initialized || typeof window === "undefined") {
    return;
  }

  SuperTokens.init({
    appInfo: {
      appName: process.env.NEXT_PUBLIC_APP_NAME || "SuperTokens Auth App",
      apiDomain: process.env.NEXT_PUBLIC_API_DOMAIN || "http://localhost:4000",
      websiteDomain: process.env.NEXT_PUBLIC_WEBSITE_DOMAIN || "http://localhost:3000",
      apiBasePath: process.env.NEXT_PUBLIC_AUTH_API_BASE_PATH || "/auth",
      websiteBasePath: process.env.NEXT_PUBLIC_AUTH_WEBSITE_BASE_PATH || "/auth"
    },
    recipeList: [
      ThirdParty.init({
        signInAndUpFeature: {
          providers: [Google.init(), Github.init()]
        }
      }),
      Session.init()
    ]
  });

  initialized = true;
}
