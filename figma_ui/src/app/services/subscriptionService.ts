import { supabase } from "../lib/supabase";

export interface Subscription {
  id: string;
  tier: string;
  status: string;
  current_period_end?: string;
}

/**
 * Get current user subscription
 */
export async function getSubscription(): Promise<Subscription | null> {
  try {
    const { data: { session } } = await supabase.auth.getSession();
    if (!session) return null;
    
    const { data, error } = await supabase
      .from('subscriptions')
      .select('*')
      .eq('user_id', session.user.id)
      .single();
      
    if (error || !data) return null;
    
    return {
      id: data.id,
      tier: data.tier,
      status: data.status,
      current_period_end: data.current_period_end
    };
  } catch (err) {
    console.error('getSubscription fallback error:', err);
    return null;
  }
}

/**
 * Stub for creating a checkout session
 */
export async function createCheckoutSession(priceId: string): Promise<{ url: string }> {
  // In the future, this would call a backend endpoint to generate a Stripe Checkout URL
  console.log(`Creating checkout session for price: ${priceId}`);
  return { url: "/settings" }; // Stub redirect to settings
}
