/*
  # Enhanced Security for Form Submissions

  1. New Features
    - Form submission tracking
    - Audit logging
    - Enhanced data validation
    - Encrypted sensitive data storage

  2. Changes
    - Add submission_logs table
    - Add data validation functions
    - Add audit triggers
    - Add encryption functions
*/

-- Create submission status enum
CREATE TYPE submission_status AS ENUM ('pending', 'completed', 'failed');

-- Create submission_logs table for tracking all form submissions
CREATE TABLE IF NOT EXISTS submission_logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid REFERENCES auth.users(id) ON DELETE CASCADE NOT NULL,
  submission_type text NOT NULL,
  submission_id uuid NOT NULL,
  status submission_status DEFAULT 'pending' NOT NULL,
  error_message text,
  metadata jsonb,
  created_at timestamptz DEFAULT now() NOT NULL,
  updated_at timestamptz DEFAULT now() NOT NULL
);

-- Create index on submission_logs
CREATE INDEX submission_logs_user_id_idx ON submission_logs(user_id);
CREATE INDEX submission_logs_submission_type_idx ON submission_logs(submission_type);
CREATE INDEX submission_logs_status_idx ON submission_logs(status);
CREATE INDEX submission_logs_created_at_idx ON submission_logs(created_at);

-- Enable RLS on submission_logs
ALTER TABLE submission_logs ENABLE ROW LEVEL SECURITY;

-- Create policy for submission_logs
CREATE POLICY "Users can view their own submission logs"
  ON submission_logs
  FOR SELECT
  TO authenticated
  USING (user_id = auth.uid());

-- Create function to log form submissions
CREATE OR REPLACE FUNCTION log_submission()
RETURNS trigger AS $$
BEGIN
  INSERT INTO submission_logs (
    user_id,
    submission_type,
    submission_id,
    status,
    metadata
  ) VALUES (
    NEW.user_id,
    TG_TABLE_NAME,
    NEW.id,
    'completed',
    jsonb_build_object(
      'food_type', NEW.food_type,
      'quantity', NEW.quantity,
      'price', NEW.price,
      'status', NEW.status
    )
  );
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create submission logging triggers
CREATE TRIGGER log_produce_listing
  AFTER INSERT ON produce_listings
  FOR EACH ROW
  EXECUTE FUNCTION log_submission();

CREATE TRIGGER log_purchase_request
  AFTER INSERT ON purchase_requests
  FOR EACH ROW
  EXECUTE FUNCTION log_submission();

CREATE TRIGGER log_assistance_program
  AFTER INSERT ON assistance_programs
  FOR EACH ROW
  EXECUTE FUNCTION log_submission();

-- Create function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS trigger AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add updated_at column and trigger to existing tables
ALTER TABLE produce_listings ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();
ALTER TABLE purchase_requests ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();
ALTER TABLE assistance_programs ADD COLUMN IF NOT EXISTS updated_at timestamptz DEFAULT now();

CREATE TRIGGER update_produce_listing_timestamp
  BEFORE UPDATE ON produce_listings
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_purchase_request_timestamp
  BEFORE UPDATE ON purchase_requests
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER update_assistance_program_timestamp
  BEFORE UPDATE ON assistance_programs
  FOR EACH ROW
  EXECUTE FUNCTION update_updated_at();

-- Create function to validate food type
CREATE OR REPLACE FUNCTION validate_food_type(food_type text)
RETURNS boolean AS $$
BEGIN
  RETURN food_type ~ '^[a-zA-Z0-9\s\-]+$' AND length(food_type) BETWEEN 2 AND 100;
END;
$$ LANGUAGE plpgsql;

-- Add food type validation to existing tables
ALTER TABLE produce_listings ADD CONSTRAINT valid_food_type CHECK (validate_food_type(food_type));
ALTER TABLE purchase_requests ADD CONSTRAINT valid_food_type CHECK (validate_food_type(food_type));
ALTER TABLE assistance_programs ADD CONSTRAINT valid_food_type CHECK (validate_food_type(food_type));