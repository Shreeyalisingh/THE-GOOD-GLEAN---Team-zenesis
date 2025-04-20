/*
  # Agricultural Marketplace Database Schema

  1. New Tables
    - produce_listings (for farmers)
    - purchase_requests (for retailers)
    - assistance_programs (for NGOs)

  2. Features
    - PostGIS integration for location data
    - Role-based access control via triggers
    - Soft delete functionality
    - Comprehensive indexing

  3. Security
    - Row Level Security (RLS) enabled
    - Role-specific policies
    - User-based access control
*/

-- Enable PostGIS extension
CREATE EXTENSION IF NOT EXISTS postgis;

-- Create enum type for status
CREATE TYPE listing_status AS ENUM ('active', 'pending', 'completed');

-- Create function to validate user role
CREATE OR REPLACE FUNCTION check_user_role(user_id uuid, required_role text)
RETURNS boolean AS $$
BEGIN
  RETURN EXISTS (
    SELECT 1 FROM profiles
    WHERE profiles.id = user_id
    AND profiles.role = required_role
  );
END;
$$ LANGUAGE plpgsql;

-- Create trigger functions for role validation
CREATE OR REPLACE FUNCTION validate_farmer()
RETURNS trigger AS $$
BEGIN
  IF NOT check_user_role(NEW.user_id, 'farmer') THEN
    RAISE EXCEPTION 'User must be a farmer';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION validate_retailer()
RETURNS trigger AS $$
BEGIN
  IF NOT check_user_role(NEW.user_id, 'retailer') THEN
    RAISE EXCEPTION 'User must be a retailer';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION validate_ngo()
RETURNS trigger AS $$
BEGIN
  IF NOT check_user_role(NEW.user_id, 'ngo') THEN
    RAISE EXCEPTION 'User must be an NGO';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create produce_listings table
CREATE TABLE IF NOT EXISTS produce_listings (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  food_type text NOT NULL,
  location geography(POINT, 4326) NOT NULL,
  price decimal(10,2) NOT NULL CHECK (price >= 0),
  quantity integer NOT NULL CHECK (quantity > 0),
  date_created timestamptz DEFAULT now() NOT NULL,
  status listing_status DEFAULT 'active' NOT NULL,
  deleted_at timestamptz
);

-- Create purchase_requests table
CREATE TABLE IF NOT EXISTS purchase_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  listing_id uuid REFERENCES produce_listings(id) ON DELETE CASCADE,
  food_type text NOT NULL,
  location geography(POINT, 4326) NOT NULL,
  price decimal(10,2) NOT NULL CHECK (price >= 0),
  quantity integer NOT NULL CHECK (quantity > 0),
  date_created timestamptz DEFAULT now() NOT NULL,
  status listing_status DEFAULT 'pending' NOT NULL,
  deleted_at timestamptz
);

-- Create assistance_programs table
CREATE TABLE IF NOT EXISTS assistance_programs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  food_type text NOT NULL,
  location geography(POINT, 4326) NOT NULL,
  price decimal(10,2) NOT NULL CHECK (price >= 0),
  quantity integer NOT NULL CHECK (quantity > 0),
  date_created timestamptz DEFAULT now() NOT NULL,
  status listing_status DEFAULT 'active' NOT NULL,
  deleted_at timestamptz
);

-- Create triggers for role validation
CREATE TRIGGER validate_farmer_role
  BEFORE INSERT OR UPDATE ON produce_listings
  FOR EACH ROW
  EXECUTE FUNCTION validate_farmer();

CREATE TRIGGER validate_retailer_role
  BEFORE INSERT OR UPDATE ON purchase_requests
  FOR EACH ROW
  EXECUTE FUNCTION validate_retailer();

CREATE TRIGGER validate_ngo_role
  BEFORE INSERT OR UPDATE ON assistance_programs
  FOR EACH ROW
  EXECUTE FUNCTION validate_ngo();

-- Enable Row Level Security
ALTER TABLE produce_listings ENABLE ROW LEVEL SECURITY;
ALTER TABLE purchase_requests ENABLE ROW LEVEL SECURITY;
ALTER TABLE assistance_programs ENABLE ROW LEVEL SECURITY;

-- Create indexes
CREATE INDEX IF NOT EXISTS produce_listings_user_id_idx ON produce_listings(user_id);
CREATE INDEX IF NOT EXISTS produce_listings_food_type_idx ON produce_listings(food_type);
CREATE INDEX IF NOT EXISTS produce_listings_status_idx ON produce_listings(status);
CREATE INDEX IF NOT EXISTS produce_listings_location_idx ON produce_listings USING GIST(location);
CREATE INDEX IF NOT EXISTS produce_listings_date_created_idx ON produce_listings(date_created);

CREATE INDEX IF NOT EXISTS purchase_requests_user_id_idx ON purchase_requests(user_id);
CREATE INDEX IF NOT EXISTS purchase_requests_listing_id_idx ON purchase_requests(listing_id);
CREATE INDEX IF NOT EXISTS purchase_requests_status_idx ON purchase_requests(status);
CREATE INDEX IF NOT EXISTS purchase_requests_location_idx ON purchase_requests USING GIST(location);
CREATE INDEX IF NOT EXISTS purchase_requests_date_created_idx ON purchase_requests(date_created);

CREATE INDEX IF NOT EXISTS assistance_programs_user_id_idx ON assistance_programs(user_id);
CREATE INDEX IF NOT EXISTS assistance_programs_food_type_idx ON assistance_programs(food_type);
CREATE INDEX IF NOT EXISTS assistance_programs_status_idx ON assistance_programs(status);
CREATE INDEX IF NOT EXISTS assistance_programs_location_idx ON assistance_programs USING GIST(location);
CREATE INDEX IF NOT EXISTS assistance_programs_date_created_idx ON assistance_programs(date_created);

-- Create RLS policies for produce_listings
CREATE POLICY "Farmers can create their own listings"
  ON produce_listings
  FOR INSERT
  TO authenticated
  WITH CHECK (
    check_user_role(auth.uid(), 'farmer')
  );

CREATE POLICY "Users can view active listings"
  ON produce_listings
  FOR SELECT
  TO authenticated
  USING (
    status = 'active'
    AND deleted_at IS NULL
  );

CREATE POLICY "Farmers can update their own listings"
  ON produce_listings
  FOR UPDATE
  TO authenticated
  USING (
    user_id = auth.uid()
    AND check_user_role(auth.uid(), 'farmer')
  );

-- Create RLS policies for purchase_requests
CREATE POLICY "Retailers can create purchase requests"
  ON purchase_requests
  FOR INSERT
  TO authenticated
  WITH CHECK (
    check_user_role(auth.uid(), 'retailer')
  );

CREATE POLICY "Users can view their own requests"
  ON purchase_requests
  FOR SELECT
  TO authenticated
  USING (
    user_id = auth.uid()
    OR EXISTS (
      SELECT 1 FROM produce_listings
      WHERE produce_listings.id = purchase_requests.listing_id
      AND produce_listings.user_id = auth.uid()
    )
  );

-- Create RLS policies for assistance_programs
CREATE POLICY "NGOs can create assistance programs"
  ON assistance_programs
  FOR INSERT
  TO authenticated
  WITH CHECK (
    check_user_role(auth.uid(), 'ngo')
  );

CREATE POLICY "Users can view active assistance programs"
  ON assistance_programs
  FOR SELECT
  TO authenticated
  USING (
    status = 'active'
    AND deleted_at IS NULL
  );

-- Create function for soft delete
CREATE OR REPLACE FUNCTION soft_delete()
RETURNS trigger AS $$
BEGIN
  NEW.deleted_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create triggers for soft delete
CREATE TRIGGER soft_delete_produce_listings
  BEFORE DELETE ON produce_listings
  FOR EACH ROW
  EXECUTE FUNCTION soft_delete();

CREATE TRIGGER soft_delete_purchase_requests
  BEFORE DELETE ON purchase_requests
  FOR EACH ROW
  EXECUTE FUNCTION soft_delete();

CREATE TRIGGER soft_delete_assistance_programs
  BEFORE DELETE ON assistance_programs
  FOR EACH ROW
  EXECUTE FUNCTION soft_delete();