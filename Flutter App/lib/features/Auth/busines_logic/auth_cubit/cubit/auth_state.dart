part of 'auth_cubit.dart';

@immutable
sealed class AuthState {}

final class AuthInitial extends AuthState {}
final class LoginStateInitial extends AuthState {}

final class LoginLoading extends AuthState {}

final class LoginSuccess extends AuthState {}

final class LoginFailure extends AuthState {
  String errorMessage;
  LoginFailure({required this.errorMessage});
}

class RegisterInitial extends AuthState {}

class RegisterLoading extends AuthState {}

class RegisterSuccess extends AuthState {}

class RegisterFailure extends AuthState {
  final String errorMessage;
  RegisterFailure({required this.errorMessage});
}

