import { serve } from 'https://deno.land/std@0.168.0/http/server.ts';
import { createClient } from 'npm:@supabase/supabase-js@2.39.7';

const corsHeaders = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type, Authorization',
};

interface ListingData {
  food_type: string;
  location: { lat: number; lng: number };
  price: number;
  quantity: number;
}

serve(async (req) => {
  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return new Response(null, { headers: corsHeaders });
  }

  try {
    const supabase = createClient(
      Deno.env.get('SUPABASE_URL') ?? '',
      Deno.env.get('SUPABASE_ANON_KEY') ?? ''
    );

    const url = new URL(req.url);
    const path = url.pathname.split('/').pop();

    // Get auth user
    const { data: { user }, error: authError } = await supabase.auth.getUser();
    if (authError || !user) {
      throw new Error('Unauthorized');
    }

    // Get user role
    const { data: profile, error: profileError } = await supabase
      .from('profiles')
      .select('role')
      .eq('id', user.id)
      .single();

    if (profileError || !profile) {
      throw new Error('User profile not found');
    }

    const role = profile.role;

    switch (req.method) {
      case 'POST': {
        const data: ListingData = await req.json();

        // Validate required fields
        if (!data.food_type || !data.location || !data.price || !data.quantity) {
          throw new Error('Missing required fields');
        }

        // Validate numeric fields
        if (data.price < 0 || data.quantity < 1) {
          throw new Error('Invalid price or quantity');
        }

        // Convert location to PostGIS point
        const point = `POINT(${data.location.lng} ${data.location.lat})`;

        let result;
        switch (role) {
          case 'farmer':
            result = await supabase
              .from('produce_listings')
              .insert({
                user_id: user.id,
                food_type: data.food_type,
                location: point,
                price: data.price,
                quantity: data.quantity,
              })
              .select()
              .single();
            break;

          case 'retailer':
            result = await supabase
              .from('purchase_requests')
              .insert({
                user_id: user.id,
                food_type: data.food_type,
                location: point,
                price: data.price,
                quantity: data.quantity,
              })
              .select()
              .single();
            break;

          case 'ngo':
            result = await supabase
              .from('assistance_programs')
              .insert({
                user_id: user.id,
                food_type: data.food_type,
                location: point,
                price: data.price,
                quantity: data.quantity,
              })
              .select()
              .single();
            break;

          default:
            throw new Error('Invalid user role');
        }

        if (result.error) {
          throw result.error;
        }

        return new Response(
          JSON.stringify(result.data),
          {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
            status: 201,
          }
        );
      }

      case 'GET': {
        let result;
        switch (role) {
          case 'farmer':
            result = await supabase
              .from('produce_listings')
              .select('*')
              .eq('user_id', user.id)
              .is('deleted_at', null)
              .order('date_created', { ascending: false });
            break;

          case 'retailer':
            result = await supabase
              .from('purchase_requests')
              .select('*')
              .eq('user_id', user.id)
              .is('deleted_at', null)
              .order('date_created', { ascending: false });
            break;

          case 'ngo':
            result = await supabase
              .from('assistance_programs')
              .select('*')
              .eq('user_id', user.id)
              .is('deleted_at', null)
              .order('date_created', { ascending: false });
            break;

          default:
            throw new Error('Invalid user role');
        }

        if (result.error) {
          throw result.error;
        }

        return new Response(
          JSON.stringify(result.data),
          {
            headers: { ...corsHeaders, 'Content-Type': 'application/json' },
          }
        );
      }

      default:
        return new Response('Method not allowed', { status: 405 });
    }
  } catch (error) {
    return new Response(
      JSON.stringify({ error: error.message }),
      {
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
        status: error.message === 'Unauthorized' ? 401 : 400,
      }
    );
  }
});