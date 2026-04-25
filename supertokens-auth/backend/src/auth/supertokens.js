const supertokens = require("supertokens-node");
const Session = require("supertokens-node/recipe/session");
const ThirdParty = require("supertokens-node/recipe/thirdparty");

const GoogleModule = require("supertokens-node/recipe/thirdparty/providers/google");
const GithubModule = require("supertokens-node/recipe/thirdparty/providers/github");

const GoogleProvider = GoogleModule.default || GoogleModule;
const GithubProvider = GithubModule.default || GithubModule;

const config = require("../config");
const { upsertUser } = require("../db/users");

function getAppInfo() {
  return {
    appName: config.appName,
    apiDomain: config.apiDomain,
    websiteDomain: config.websiteDomain,
    apiBasePath: config.apiBasePath,
    websiteBasePath: config.websiteBasePath
  };
}

function initSuperTokens() {
  supertokens.init({
    framework: "express",
    supertokens: {
      connectionURI: config.supertokensConnectionURI,
      ...(config.supertokensApiKey ? { apiKey: config.supertokensApiKey } : {})
    },
    appInfo: getAppInfo(),
    recipeList: [
      ThirdParty.init({
        signInAndUpFeature: {
          providers: [
            GoogleProvider({
              clientId: config.googleClientId,
              clientSecret: config.googleClientSecret
            }),
            GithubProvider({
              clientId: config.githubClientId,
              clientSecret: config.githubClientSecret
            })
          ]
        },
        override: {
          functions: (originalImplementation) => ({
            ...originalImplementation,
            signInUp: async function signInUp(input) {
              const response = await originalImplementation.signInUp(input);

              if (response.status === "OK") {
                const user = response.user;
                const email = user.emails?.[0] || null;

                try {
                  await upsertUser({
                    supertokensUserId: user.id,
                    email,
                    provider: input.thirdPartyId
                  });
                } catch (error) {
                  // Keep auth flow healthy even if app DB write fails temporarily.
                  console.error("Failed to upsert user into Neon:", error);
                }
              }

              return response;
            }
          })
        }
      }),
      Session.init({
        cookieSecure: config.nodeEnv === "production",
        cookieSameSite: config.nodeEnv === "production" ? "none" : "lax",
        exposeAccessTokenToFrontendInCookieBasedAuth: false,
        antiCsrf: "VIA_TOKEN"
      })
    ]
  });
}

module.exports = {
  initSuperTokens,
  getAppInfo
};
