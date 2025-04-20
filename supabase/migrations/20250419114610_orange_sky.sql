/*
  # Enhanced Role-Based Access Control

  1. Security Updates
    - Add RLS policies for role-based access
    - Add function to validate user roles
    - Add trigger for role validation

  2. Changes
    - Add role validation function
    - Update existing RLS policies
    - Add new policies for role-specific access
*/

-- Function to validate user role
CREATE OR REPLACE FUNCTION check_user_role(required_role text)
RETURNS boolean AS $$
BEGIN
  RETURN (
    SELECT role = required_role
    FROM profiles
    WHERE id = auth.uid()
  );
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Update existing policies with role checks
DROP POLICY IF EXISTS "Users can read their own profile" ON profiles;
CREATE POLICY "Users can read their own profile"
  ON profiles
  FOR SELECT
  TO authenticated
  USING (
    id = auth.uid()
    AND (
      role = 'farmer'
      OR role = 'retailer'
      OR role = 'ngo'
    )
  );

DROP POLICY IF EXISTS "Users can update their own profile" ON profiles;
CREATE POLICY "Users can update their own profile"
  ON profiles
  FOR UPDATE
  TO authenticated
  USING (
    id = auth.uid()
    AND (
      role = 'farmer'
      OR role = 'retailer'
      OR role = 'ngo'
    )
  )
  WITH CHECK (
    id = auth.uid()
    AND role = (SELECT role FROM profiles WHERE id = auth.uid())
  );

-- Create function to ensure role cannot be changed after creation
CREATE OR REPLACE FUNCTION prevent_role_update()
RETURNS TRIGGER AS $$
BEGIN
  IF OLD.role <> NEW.role THEN
    RAISE EXCEPTION 'Role cannot be changed after creation';
  END IF;
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to prevent role updates
DROP TRIGGER IF EXISTS ensure_role_immutable ON profiles;
CREATE TRIGGER ensure_role_immutable
  BEFORE UPDATE ON profiles
  FOR EACH ROW
  EXECUTE FUNCTION prevent_role_update();

-- Add index for role-based queries if not exists
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_indexes
    WHERE tablename = 'profiles'
    AND indexname = 'profiles_role_idx'
  ) THEN
    CREATE INDEX profiles_role_idx ON profiles(role);
  END IF;
END $$;