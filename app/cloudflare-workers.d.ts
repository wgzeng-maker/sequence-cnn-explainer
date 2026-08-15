declare module "cloudflare:workers" {
  interface D1PreparedStatement {
    bind(...values: unknown[]): D1PreparedStatement;
    first<T>(): Promise<T | null>;
    run(): Promise<unknown>;
  }

  interface D1Database {
    prepare(query: string): D1PreparedStatement;
  }

  export const env: { DB: D1Database };
}
