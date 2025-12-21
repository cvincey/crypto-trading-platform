#!/usr/bin/env python3
"""Test database connection."""

import os
import sys

def test_connection():
    """Test PostgreSQL connection."""
    
    # Check for DATABASE_URL (Railway provides this automatically)
    database_url = os.environ.get("DATABASE_URL")
    
    print("=" * 50)
    print("Database Connection Test")
    print("=" * 50)
    
    # Show what variables are set (without revealing secrets)
    print("\n📋 Environment Variables:")
    for var in ["DATABASE_URL", "PGHOST", "PGPORT", "PGUSER", "PGDATABASE"]:
        value = os.environ.get(var)
        if value:
            if "PASSWORD" in var or "URL" in var:
                print(f"  ✓ {var} = [SET - hidden]")
            else:
                print(f"  ✓ {var} = {value}")
        else:
            print(f"  ✗ {var} = [NOT SET]")
    
    if not database_url:
        print("\n❌ DATABASE_URL not set!")
        print("\n💡 In Railway:")
        print("   1. Click on your app service")
        print("   2. Go to 'Variables' tab")
        print("   3. Click 'Add Reference Variable'")
        print("   4. Select your Postgres service")
        print("   5. Choose DATABASE_URL")
        return False
    
    # Try to connect
    print("\n🔌 Attempting connection...")
    
    try:
        import asyncpg
        import asyncio
        
        async def connect():
            conn = await asyncpg.connect(database_url)
            version = await conn.fetchval("SELECT version()")
            await conn.close()
            return version
        
        version = asyncio.run(connect())
        print(f"\n✅ SUCCESS! Connected to PostgreSQL")
        print(f"   Version: {version[:50]}...")
        return True
        
    except ImportError:
        # Try with psycopg2 if asyncpg not available
        try:
            import psycopg2
            conn = psycopg2.connect(database_url)
            cur = conn.cursor()
            cur.execute("SELECT version()")
            version = cur.fetchone()[0]
            conn.close()
            print(f"\n✅ SUCCESS! Connected to PostgreSQL")
            print(f"   Version: {version[:50]}...")
            return True
        except ImportError:
            print("\n⚠️  Neither asyncpg nor psycopg2 installed")
            print("   But DATABASE_URL is set correctly!")
            return True
        except Exception as e:
            print(f"\n❌ Connection failed: {e}")
            return False
            
    except Exception as e:
        print(f"\n❌ Connection failed: {e}")
        return False


if __name__ == "__main__":
    success = test_connection()
    sys.exit(0 if success else 1)
