//! Tiny fuel-metered stack VM — the reducer sandbox.
//!
//! Stands in for a wasm module (wasmtime in the reference architecture). The
//! critical property is the same: every instruction charges fuel against a
//! per-invocation budget, so a runaway reducer traps with OutOfFuel and its
//! transaction rolls back cleanly instead of stalling the engine.

use storage::{Txn, Value};

#[derive(Clone, Debug)]
pub enum Instr {
    PushInt(i64),
    Pop,
    /// Duplicate the top of stack.
    Dup,
    Add,
    Sub,
    Mul,
    /// Load the Int value at key onto the stack (0 if absent / non-int).
    LoadInt(Vec<u8>),
    /// Pop an Int and store it at key.
    StoreInt(Vec<u8>),
    /// Unconditional jump to instruction index.
    Jump(usize),
    /// Pop; if zero, jump to index.
    JumpIfZero(usize),
    /// Pop the top of stack and return it as the reducer output.
    Return,
    /// Explicit trap (test/guard).
    Trap(String),
}

#[derive(Clone, Debug)]
pub struct Program {
    pub instrs: Vec<Instr>,
}

#[derive(Debug)]
pub enum VmError {
    OutOfFuel,
    StackUnderflow,
    BadJump(usize),
    Trap(String),
}

impl std::fmt::Display for VmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            VmError::OutOfFuel => write!(f, "reducer aborted: out of fuel"),
            VmError::StackUnderflow => write!(f, "reducer aborted: stack underflow"),
            VmError::BadJump(i) => write!(f, "reducer aborted: bad jump to {i}"),
            VmError::Trap(m) => write!(f, "reducer trap: {m}"),
        }
    }
}
impl std::error::Error for VmError {}

pub struct Vm {
    stack: Vec<i64>,
    fuel: u64,
    fuel_budget: u64,
}

impl Vm {
    pub fn new(fuel_budget: u64) -> Self {
        Self { stack: Vec::new(), fuel: fuel_budget, fuel_budget }
    }

    pub fn fuel_used(&self) -> u64 {
        self.fuel_budget - self.fuel
    }

    fn burn(&mut self, cost: u64) -> Result<(), VmError> {
        if self.fuel < cost {
            return Err(VmError::OutOfFuel);
        }
        self.fuel -= cost;
        Ok(())
    }

    fn pop(&mut self) -> Result<i64, VmError> {
        self.stack.pop().ok_or(VmError::StackUnderflow)
    }

    /// Run the program against a transaction. Returns Some(output) on Return.
    pub fn run(&mut self, prog: &Program, txn: &mut Txn) -> Result<Option<i64>, VmError> {
        let mut pc = 0usize;
        loop {
            self.burn(1)?; // every instruction costs at least 1 fuel
            let instr = prog.instrs.get(pc).ok_or(VmError::BadJump(pc))?;
            match instr {
                Instr::PushInt(v) => {
                    self.stack.push(*v);
                    pc += 1;
                }
                Instr::Pop => {
                    self.pop()?;
                    pc += 1;
                }
                Instr::Dup => {
                    let v = *self.stack.last().ok_or(VmError::StackUnderflow)?;
                    self.stack.push(v);
                    pc += 1;
                }
                Instr::Add => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(a.wrapping_add(b));
                    pc += 1;
                }
                Instr::Sub => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(a.wrapping_sub(b));
                    pc += 1;
                }
                Instr::Mul => {
                    let b = self.pop()?;
                    let a = self.pop()?;
                    self.stack.push(a.wrapping_mul(b));
                    pc += 1;
                }
                Instr::LoadInt(key) => {
                    self.burn(10)?; // storage access costs more fuel
                    let v = match txn.get(key) {
                        Some(Value::Int(i)) => i,
                        _ => 0,
                    };
                    self.stack.push(v);
                    pc += 1;
                }
                Instr::StoreInt(key) => {
                    self.burn(10)?;
                    let v = self.pop()?;
                    txn.put(key.clone(), Value::Int(v));
                    pc += 1;
                }
                Instr::Jump(t) => {
                    pc = *t;
                }
                Instr::JumpIfZero(t) => {
                    let v = self.pop()?;
                    if v == 0 {
                        pc = *t;
                    } else {
                        pc += 1;
                    }
                }
                Instr::Return => {
                    return Ok(self.stack.pop());
                }
                Instr::Trap(m) => {
                    return Err(VmError::Trap(m.clone()));
                }
            }
        }
    }
}
